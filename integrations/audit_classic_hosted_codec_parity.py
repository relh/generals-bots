"""Compare Classic training and hosted public codecs on verified replay states."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from generals.core import game
from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import encode_wire_observation


ENGINE_SHA256 = "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"
FRAME_FIELDS = ("turn", "type_grid", "owner_grid", "army_grid", "army", "land")
SPATIAL_FIELDS = ("armies", "ownership_neutral", "generals", "castles", "mountains", "passable")


def padded_training_state(state):
    height, width = state.armies.shape
    if not (18 <= height <= 21 and 18 <= width <= 21):
        raise ValueError("Expected a hosted Classic 18–21 tile rectangle")
    padding = ((0, 21 - height), (0, 21 - width))
    values = {}
    for name in state._fields:
        value = np.asarray(getattr(state, name))
        if name == "ownership":
            value = np.pad(value, ((0, 0), *padding))
        elif name in SPATIAL_FIELDS:
            value = np.pad(value, padding, constant_values=name == "mountains")
        values[name] = jnp.asarray(value)
    return game.GameState(**values)


def compare_state(match):
    padded = padded_training_state(match.state)
    maximum = 0.0
    for side in (0, 1):
        hosted_values, hosted_mask = encode_wire_observation(
            match.observation(side), directional=True, factorized_actions=False,
            public_scalar_features=True,
        )
        training_observation = game.get_observation(padded, side)
        training_values, training_mask = encode_coworld_directional_observation(
            training_observation, factorized_actions=False, public_scalar_features=True,
        )
        training_values = np.asarray(training_values, np.float32)
        training_mask = np.asarray(training_mask, bool)
        if hosted_values.shape != (7056,) or training_values.shape != (7056,):
            raise ValueError("Classic public scalar codec has the wrong shape")
        if not np.array_equal(hosted_mask, training_mask):
            raise ValueError(f"Hosted action mask differs on turn {match.turn}, side {side}")
        difference = float(np.max(np.abs(hosted_values - training_values)))
        if difference > 1e-7:
            raise ValueError(f"Hosted observation differs on turn {match.turn}, side {side}: {difference}")
        maximum = max(maximum, difference)
    return maximum


def audit(replay_root, game_indices, turns):
    engine = Path(__file__).resolve().parents[1] / "generals/core/coworld_game.py"
    engine_sha256 = hashlib.sha256(engine.read_bytes()).hexdigest()
    if engine_sha256 != ENGINE_SHA256:
        raise ValueError("Local Classic engine differs from the pinned hosted revision")
    manifest = json.loads((replay_root / "leader-replay-manifest.json").read_text())
    results = []
    for index in game_indices:
        item = manifest[index]
        blob = (replay_root / "leader-replays" / (item["episode_id"] + ".bin")).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Replay checksum differs: game {index}")
        replay = json.loads(gzip.decompress(blob))
        if replay["ruleset"] != "classic":
            raise ValueError(f"Replay {index} is not Classic")
        match = Match(replay["seed"], coworld_classic_rules=True)
        checked, maximum = [], 0.0
        for turn_index, receipt in enumerate(replay["turns"]):
            if turn_index > max(turns):
                break
            frame = match.frame()
            if not receipt["applied"] or receipt["turn"] != match.turn or any(
                frame[field] != replay["frames"][turn_index][field]
                for field in FRAME_FIELDS
            ):
                raise ValueError(f"Classic replay frame differs: game {index} turn {turn_index}")
            if turn_index in turns:
                maximum = max(maximum, compare_state(match))
                checked.append(turn_index)
            match.advance(receipt["actions"])
        if not checked:
            raise ValueError(f"No requested frame exists in game {index}")
        results.append(dict(game=index, board=list(match.state.armies.shape), turns=checked,
                            seats=2, max_absolute_value_difference=maximum))
    return dict(engine_sha256=engine_sha256, games=len(results),
                checked_public_states=2 * sum(len(row["turns"]) for row in results),
                max_absolute_value_difference=max(row["max_absolute_value_difference"] for row in results),
                action_masks_equal=True, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--games", type=int, nargs="+", default=[0, 6, 7, 10])
    parser.add_argument("--turns", type=int, nargs="+", default=[0, 25, 99, 100, 150, 200])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.games or not args.turns or min(args.games) < 0 or min(args.turns) < 0:
        raise ValueError("Select nonnegative replay and turn indices")
    result = audit(args.replay_root, args.games, set(args.turns))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

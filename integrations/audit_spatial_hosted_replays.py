"""Compare exported spatial policies on verified public states in hosted replays.

Only the prefix whose frames match the local Classic engine is scored. Replay
actions are labels for this diagnostic; no opponent weights are required.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import training_observation
from integrations.spatial_policy_bundle import SpatialPlayerPolicy, structured_action_probabilities


FRAME_FIELDS = ("turn", "type_grid", "owner_grid", "army_grid", "army", "land")
encode = jax.jit(lambda obs: encode_coworld_directional_observation(
    obs, factorized_actions=False, public_scalar_features=True))


def flat_action(action):
    passed, row, col, direction, half = map(int, action)
    return 3528 if passed else (half * 4 + direction) * 441 + row * 21 + col


def verified_states(replay_root, limit):
    manifest = json.loads((replay_root / "leader-replay-manifest.json").read_text())
    observations, masks, actions, games, divergences = [], [], [], [], []
    for game_index, item in enumerate(manifest[:limit]):
        blob = (replay_root / "leader-replays" / (item["episode_id"] + ".bin")).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Replay checksum differs: game {game_index}")
        replay = json.loads(gzip.decompress(blob))
        if replay["ruleset"] != "classic":
            raise ValueError(f"Expected Classic: game {game_index}")
        match = Match(replay["seed"])
        if any(match.frame()[field] != replay["frames"][0][field] for field in FRAME_FIELDS):
            raise ValueError(f"Initial frame differs: game {game_index}")
        expert = 1 - item["seat"]
        for turn_index, receipt in enumerate(replay["turns"]):
            if not receipt["applied"] or receipt["turn"] != match.turn:
                raise ValueError(f"Turn receipt differs: game {game_index} turn {turn_index}")
            values, mask = encode(training_observation(match.observation(expert)))
            action = flat_action(receipt["actions"][expert])
            if not bool(mask[action]):
                raise ValueError(f"Recorded expert action is masked: game {game_index} turn {turn_index}")
            observations.append(np.asarray(values, np.float32))
            masks.append(np.asarray(mask, bool))
            actions.append(action)
            games.append(game_index)
            match.advance(receipt["actions"])
            differing = [field for field in FRAME_FIELDS
                         if match.frame()[field] != replay["frames"][turn_index + 1][field]]
            if differing:
                # A captured general can leave different postgame army counts.
                if turn_index == len(replay["turns"]) - 1 and set(differing) <= {"army_grid", "army"}:
                    continue
                divergences.append({"game": game_index, "turn": turn_index,
                                    "fields": differing, "verified_actions": turn_index + 1,
                                    "total_turns": len(replay["turns"])})
                break
    return (np.asarray(observations), np.asarray(masks), np.asarray(actions),
            np.asarray(games), divergences)


def score(bundle, observations, masks, actions, games, move_temperature=None):
    policy = SpatialPlayerPolicy(bundle)
    if policy.observation_size != observations.shape[1] or policy.action_mode != "structured_sample":
        raise ValueError("Expected 16-plane directional structured-sampling policy")
    if move_temperature is None:
        move_temperature = policy.move_temperature
    if not np.isfinite(move_temperature) or move_temperature <= 0:
        raise ValueError("Move temperature must be finite and positive")
    logprob, route_match, action_match = [], [], []
    action_entropy, route_entropy, half_mass, top_mass = [], [], [], []
    for start in range(0, len(actions), 32):
        stop = min(start + 32, len(actions))
        outputs = policy.forward(observations[start:stop])
        for index, output in enumerate(outputs, start):
            probabilities = structured_action_probabilities(
                output, masks[index], move_temperature, policy.split_temperature,
                observations=observations[index], neutral_route_bias=policy.neutral_route_bias,
                weak_owned_route_penalty=policy.weak_owned_route_penalty,
                doomed_attack_route_penalty=policy.doomed_attack_route_penalty)
            action = actions[index]
            predicted = int(np.argmax(probabilities))
            logprob.append(float(np.log(probabilities[action])))
            route_match.append(int(predicted == action if action == 3528 or predicted == 3528
                                   else predicted % 1764 == action % 1764))
            action_match.append(int(predicted == action))
            present = probabilities[probabilities > 0]
            action_entropy.append(float(-np.sum(present * np.log(present))))
            routes = np.concatenate((probabilities[:1764] + probabilities[1764:3528],
                                     probabilities[3528:]))
            present_routes = routes[routes > 0]
            route_entropy.append(float(-np.sum(present_routes * np.log(present_routes))))
            half_mass.append(float(np.sum(probabilities[1764:3528])))
            top_mass.append(float(probabilities[predicted]))
    logprob = np.asarray(logprob)
    per_game = {str(game): {"actions": int(np.sum(games == game)),
                             "nll": float(-logprob[games == game].mean())}
                for game in np.unique(games)}
    return {"policy_sha256": hashlib.sha256((bundle / "policy.bin").read_bytes()).hexdigest(),
            "move_temperature": move_temperature, "bundle_move_temperature": policy.move_temperature,
            "actions": len(actions), "nll": float(-logprob.mean()),
            "route_match": float(np.mean(route_match)), "action_match": float(np.mean(action_match)),
            "action_entropy": float(np.mean(action_entropy)), "route_entropy": float(np.mean(route_entropy)),
            "half_probability": float(np.mean(half_mass)), "top_probability": float(np.mean(top_mass)),
            "per_game": per_game, "logprob": logprob.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--move-temperature", type=float, action="append",
                        help="Counterfactual route temperature; repeat to compare without changing a bundle")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observations, masks, actions, games, divergences = verified_states(args.replay_root, args.limit)
    result = {"replay_root": str(args.replay_root), "games": len(np.unique(games)),
              "verified_actions": len(actions), "divergences": divergences,
              "expert_half_actions": int(np.sum((actions >= 1764) & (actions < 3528))),
              "expert_pass_actions": int(np.sum(actions == 3528)), "policies": {}}
    for bundle in args.bundle:
        for temperature in args.move_temperature or [None]:
            key = str(bundle) if temperature is None else f"{bundle}@routeT={temperature:g}"
            result["policies"][key] = score(bundle, observations, masks, actions, games, temperature)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "policies"}))
    for bundle, metrics in result["policies"].items():
        print(json.dumps({"bundle": bundle, **{key: value for key, value in metrics.items()
                                                if key not in ("logprob", "per_game")}}))


if __name__ == "__main__":
    main()

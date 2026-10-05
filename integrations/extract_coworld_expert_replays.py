"""Reconstruct public Coworld observations and actions from hosted replays.

The archived opponent policy is never queried. Each label comes from the
recorded action, and each observation is computed before that action executes.
"""

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

import jax
import numpy as np

from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import training_observation


BOARD_SIZE = 21
ACTION_COUNT = 8 * BOARD_SIZE * BOARD_SIZE + 1
FRAME_FIELDS = ("turn", "type_grid", "owner_grid", "army_grid", "army", "land")
encode_flat = jax.jit(lambda obs: encode_coworld_directional_observation(obs, factorized_actions=False))


def flat_action(action):
    passed, row, col, direction, half = map(int, action)
    if passed:
        return ACTION_COUNT - 1
    return half * 4 * BOARD_SIZE**2 + direction * BOARD_SIZE**2 + row * BOARD_SIZE + col


def extract(archive_path: Path, output_path: Path, limit: int | None = None, both_seats: bool = False):
    samples = {key: [] for key in ("observations", "action_masks", "actions", "episode", "turn", "seat")}
    episodes = []
    with tarfile.open(archive_path, "r:gz") as archive:
        manifest_member = next(m for m in archive if
                               (m.name.endswith("replay-analysis-20260928.json")
                                or m.name.endswith("expert-manifest.json"))
                               and not Path(m.name).name.startswith("._"))
        manifest = json.load(archive.extractfile(manifest_member))
        records = sorted(manifest["records"], key=lambda r: (r.get("seat", 0), r["episode_index"]))
        if limit is not None:
            records = records[:limit]
        for record in records:
            member = archive.getmember(Path(record["replay_path"]).name)
            replay = json.loads(gzip.decompress(archive.extractfile(member).read()))
            if replay["ruleset"] != "classic":
                raise ValueError(f"Unexpected ruleset: {member.name}")
            match = Match(replay["seed"])
            if any(match.frame()[field] != replay["frames"][0][field] for field in FRAME_FIELDS):
                raise ValueError(f"Initial state differs: {member.name}")
            expert_seats = (0, 1) if both_seats else (1 - record["seat"],)
            start = len(samples["actions"])
            for turn_index, receipt in enumerate(replay["turns"]):
                if not receipt["applied"] or receipt["turn"] != match.turn:
                    raise ValueError(f"Turn receipt differs: {member.name} turn {turn_index}")
                for expert_seat in expert_seats:
                    public = match.observation(expert_seat)
                    obs = training_observation(public)
                    values, mask = encode_flat(obs)
                    action = flat_action(receipt["actions"][expert_seat])
                    if not 0 <= action < ACTION_COUNT or not bool(mask[action]):
                        raise ValueError(f"Expert action is masked: {member.name} turn {turn_index} seat {expert_seat}")
                    samples["observations"].append(np.asarray(values, dtype=np.float16))
                    samples["action_masks"].append(np.packbits(np.asarray(mask, dtype=np.uint8)))
                    samples["actions"].append(action)
                    samples["episode"].append(len(episodes))
                    samples["turn"].append(match.turn)
                    samples["seat"].append(expert_seat)
                match.advance(receipt["actions"])
                frame = match.frame()
                if any(frame[field] != replay["frames"][turn_index + 1][field] for field in FRAME_FIELDS):
                    raise ValueError(f"Replay frame differs: {member.name} turn {turn_index}")
            episodes.append({
                "member": member.name, "seed": replay["seed"],
                "expert_seat": "both" if both_seats else expert_seats[0],
                "candidate_seat": record.get("seat"), "start": start, "end": len(samples["actions"]),
                "split": record.get("split", "holdout" if record["episode_index"] >= 6 else "train"),
            })
            print(json.dumps(episodes[-1]), flush=True)
    metadata = {
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "expert_policy_version_id": manifest["daveey_policy_version_id"],
        "observation": "hint-free directional 11 x 21 x 21 public planes, float16",
        "action": "flat 8 x 21 x 21 plus pass; split is high block",
        "mask": "np.packbits over 3529 boolean actions",
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **{k: np.asarray(v) for k, v in samples.items()},
                        metadata=np.asarray(json.dumps(metadata)))
    print(json.dumps({"output": str(output_path), "samples": len(samples["actions"]),
                      "episodes": len(episodes), "train_episodes": sum(e["split"] == "train" for e in episodes)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--both-seats", action="store_true")
    args = parser.parse_args()
    extract(args.archive, args.output, args.limit, args.both_seats)

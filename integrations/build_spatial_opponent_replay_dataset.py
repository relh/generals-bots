"""Build verified public-state shards for an opponent-only replay model.

The learner must not use these action labels as teacher targets. A fitted
opponent can instead join a frozen self-play pool and provide RL competition.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from integrations.audit_spatial_hosted_replays import verified_states


def game_split(seed: int, holdout_percent: int) -> str:
    digest = hashlib.sha256(f"classic-opponent:{seed}".encode()).digest()
    return "validation" if int.from_bytes(digest[:8], "little") % 100 < holdout_percent else "train"


def build(replay_roots: list[Path], output: Path, opponent_id: str, holdout_percent: int) -> dict:
    if not replay_roots or not 0 < holdout_percent < 100:
        raise ValueError("Require replay roots and a holdout percentage between 1 and 99")
    if output.exists():
        raise FileExistsError(output)
    seen_seeds: set[int] = set()
    output.mkdir(parents=True)
    record = {"schema": "coworld-classic-opponent-replay-v1", "opponent_policy_version_id": opponent_id,
              "holdout_percent": holdout_percent, "shards": [], "games": 0, "actions": 0,
              "train_games": 0, "validation_games": 0,
              "train_actions": 0, "validation_actions": 0}
    for index, root in enumerate(replay_roots):
        manifest_path = root / "leader-replay-manifest.json"
        entries = json.loads(manifest_path.read_text())
        if not entries:
            raise ValueError(f"Empty replay manifest: {manifest_path}")
        seeds, seats, splits, turns = [], [], [], []
        for entry in entries:
            if entry.get("opponent_policy_version_ids") != [opponent_id]:
                raise ValueError(f"Wrong opponent in {manifest_path}: {entry['episode_id']}")
            blob = (root / "leader-replays" / (entry["episode_id"] + ".bin")).read_bytes()
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError(f"Replay SHA256 mismatch: {entry['episode_id']}")
            replay = json.loads(gzip.decompress(blob))
            if replay["ruleset"] != "classic":
                raise ValueError(f"Wrong ruleset: {entry['episode_id']}")
            seed = int(replay["seed"])
            if seed in seen_seeds:
                raise ValueError(f"Duplicate game seed: {seed}")
            seen_seeds.add(seed)
            seeds.append(seed)
            seats.append(1 - int(entry["seat"]))
            splits.append(game_split(seed, holdout_percent))
            turns.append(len(replay["turns"]))
        observations, masks, actions, games, divergences = verified_states(root, len(entries))
        if divergences:
            raise ValueError(f"Classic replay divergence in {root}: {divergences}")
        if not (len(observations) == len(masks) == len(actions) == len(games) == sum(turns)):
            raise ValueError(f"Replay action counts differ: {root}")
        if observations.shape[1:] != (16 * 441,) or masks.shape[1:] != (3529,):
            raise ValueError(f"Unexpected public observation or mask shape: {root}")
        if len(entries) > np.iinfo(np.uint16).max or actions.max() >= 3529:
            raise ValueError(f"Replay shard exceeds compact index range: {root}")
        shard = output / f"shard-{index:03d}.npz"
        np.savez_compressed(shard,
                            observations=observations.astype(np.float16),
                            mask_bits=np.packbits(masks, axis=1, bitorder="little"),
                            actions=actions.astype(np.uint16),
                            game_indices=games.astype(np.uint16),
                            seeds=np.asarray(seeds, np.uint32),
                            expert_seats=np.asarray(seats, np.uint8),
                            validation_games=np.asarray([split == "validation" for split in splits], bool))
        split_actions = np.bincount(games, minlength=len(entries))
        for split in ("train", "validation"):
            chosen = np.asarray([part == split for part in splits])
            record[f"{split}_games"] += int(chosen.sum())
            record[f"{split}_actions"] += int(split_actions[chosen].sum())
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        record["shards"].append({"file": shard.name, "sha256": digest,
                                 "replay_root": str(root), "replay_manifest_sha256":
                                 hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                 "games": len(entries), "actions": len(actions),
                                 "expert_half_actions": int(((actions >= 1764) & (actions < 3528)).sum())})
        record["games"] += len(entries)
        record["actions"] += len(actions)
        print(f"{shard.name}: {len(entries)} exact games, {len(actions)} public actions, SHA256 {digest}", flush=True)
    (output / "dataset.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opponent-policy-version-id", required=True)
    parser.add_argument("--holdout-percent", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(build(args.replay_root, args.output, args.opponent_policy_version_id,
                           args.holdout_percent), indent=2))


if __name__ == "__main__":
    main()

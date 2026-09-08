"""Compare complete paired arena runs, resampling independent maps, not games."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def indexed_rows(directory):
    rows = list(csv.DictReader((directory / "games.csv").open()))
    indexed = {}
    for row in rows:
        if row["result"] not in ("win", "loss", "draw"):
            raise ValueError("unfinished/unknown outcomes cannot enter strength comparisons")
        key = tuple(row.get(k, "") for k in ("suite", "opponent", "board_id", "repeat", "swapped", "seat"))
        if key in indexed:
            raise ValueError(f"duplicate case: {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError("empty run")
    return indexed


def compare(control, candidate, resamples=100000):
    control, candidate = Path(control), Path(candidate)
    left, right = indexed_rows(control), indexed_rows(candidate)
    if left.keys() != right.keys():
        raise ValueError("runs must contain identical case keys")
    metadata = [json.loads((p / "metadata.json").read_text()) for p in (control, candidate)]
    # Compare every retained board array, rather than trusting seed labels alone.
    names = sorted(p.name for p in control.glob("*boards.npz"))
    if not names or names != sorted(p.name for p in candidate.glob("*boards.npz")):
        raise ValueError("missing or different board archives")
    for name in names:
        with np.load(control / name, allow_pickle=False) as a, np.load(candidate / name, allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError(f"different boards: {name}")
    for field in ("suite_rules", "rules"):
        if metadata[0].get(field) != metadata[1].get(field):
            raise ValueError(f"different {field}")
    if isinstance(metadata[0].get("opponent"), dict):
        opponent_hashes = [m.get("opponent", {}).get("directory_sha256") for m in metadata]
        if not opponent_hashes[0] or opponent_hashes[0] != opponent_hashes[1]:
            raise ValueError("different or missing external opponent identity")
    sources = [m.get("source_hashes", m.get("source_sha256", {})) for m in metadata]
    required = {
        "generals/core/game.py",
        "generals/core/action.py",
        "generals/core/observation.py",
        "generals/core/grid.py",
        "generals/modifiers/build_castles.py",
        "generals/modifiers/deathtouch.py",
        "generals/evaluation/arena.py",
    }
    external = isinstance(metadata[0].get("opponent"), dict)
    required.add("scripts/stdio_arena.py" if external else "generals/evaluation/cli.py")
    if not external:
        required.add("generals/agents/agent.py")
        for name in {key[1] for key in left}:
            module = "sentinel_v3" if name.startswith("sentinel-v3") else name
            required.add(f"generals/agents/{module}_agent.py")
            if module == "sentinel_v3":
                required.add("generals/agents/sentinel_agent.py")
    if any(not required <= source.keys() or any(not source[k] for k in required) for source in sources):
        raise ValueError("missing simulator, runner, or opponent source identity")
    physics = [
        {k: v for k, v in source.items() if k.startswith(("generals/core/", "generals/modifiers/"))}
        for source in sources
    ]
    if physics[0] != physics[1] or any(sources[0][k] != sources[1][k] for k in required):
        raise ValueError("different simulator, runner, or opponent sources")
    if metadata[0].get("opponent_options", {}) != metadata[1].get("opponent_options", {}):
        raise ValueError("different opponent options")
    groups = {}
    values = {"win": 1.0, "loss": 0.0, "draw": 0.5}
    for key, old in left.items():
        new = right[key]
        if old.get("action_seed") != new.get("action_seed"):
            raise ValueError(f"different action seed: {key}")
        group = groups.setdefault(key[:2], {})
        group.setdefault(key[2], []).append((old, new))
    results = {}
    rng = np.random.default_rng(19983)
    for group_key, maps in groups.items():
        # Each map must contain all seat/label assignments for every repeat.
        for board, pairs in maps.items():
            cases = {(p[0]["repeat"], p[0]["swapped"], p[0]["seat"]) for p in pairs}
            repeats = {p[0]["repeat"] for p in pairs}
            if cases != {(r, s, p) for r in repeats for s in ("0", "1") for p in ("0", "1")}:
                raise ValueError(f"incomplete map cluster: {group_key}/{board}")
        map_scores = np.array(
            [[np.mean([values[p[i]["result"]] for p in pairs]) for i in (0, 1)] for pairs in maps.values()]
        )
        delta = map_scores[:, 1] - map_scores[:, 0]
        bootstrap = rng.choice(delta, (resamples, len(delta)), replace=True).mean(axis=1)
        all_pairs = [pair for pairs in maps.values() for pair in pairs]
        results["/".join(group_key).strip("/") or "external"] = {
            "maps": len(maps),
            "games_per_candidate": len(all_pairs),
            "control_score": float(map_scores[:, 0].mean()),
            "candidate_score": float(map_scores[:, 1].mean()),
            "paired_score_difference": float(delta.mean()),
            "paired_score_difference_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "counts": {
                role: {outcome: sum(p[i]["result"] == outcome for p in all_pairs) for outcome in values}
                for i, role in enumerate(("control", "candidate"))
            },
            "board_score_differences": dict(zip(maps, delta.tolist())),
        }
    return {
        "control": str(control),
        "candidate": str(candidate),
        "resamples": resamples,
        "method": "paired map-cluster percentile bootstrap; development comparisons are exploratory",
        "games_sha256": [hashlib.sha256((p / "games.csv").read_bytes()).hexdigest() for p in (control, candidate)],
        "metadata_sha256": [
            hashlib.sha256((p / "metadata.json").read_bytes()).hexdigest() for p in (control, candidate)
        ],
        "matchups": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["matchups"], indent=2))


if __name__ == "__main__":
    main()

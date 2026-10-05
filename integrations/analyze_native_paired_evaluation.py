"""Verify paired evaluation records and bootstrap map-cluster score contrasts."""

import argparse
import json
from pathlib import Path

import numpy as np


def contrast(root, baseline, candidate, seed, resamples):
    rng = np.random.default_rng(seed)
    reports = {}
    for opponent in ("expander_harvester", "sentinel", "strong_mixed"):
        before = root / f"{baseline}-{opponent}"
        after = root / f"{candidate}-{opponent}"
        records = [json.loads((p / "evaluation.json").read_text()) for p in (before, after)]
        for record in records:
            assert record["held_out"] and record["action_selection"] == "argmax_per_head"
        for key in ("games", "seed", "reset_seed", "opponent", "options"):
            assert records[0][key] == records[1][key], key
        for name in ("initial_state_sha256", "initial_sides", "initial_opponent_ids"):
            assert np.array_equal(np.load(before / f"{name}.npy"), np.load(after / f"{name}.npy"))
        hashes = np.load(before / "initial_state_sha256.npy")
        outcomes = [np.load(p / "outcomes.npy") for p in (before, after)]
        for record, scores in zip(records, outcomes, strict=True):
            assert scores.shape == hashes.shape == (record["games"],)
            assert np.isin(scores, (-1, 0, 1)).all()
            assert [int((scores == value).sum()) for value in (1, -1, 0)] == [
                record["wins"], record["losses"], record["draws"]
            ]
        _, groups = np.unique(hashes, return_inverse=True)
        difference = outcomes[1] - outcomes[0]
        sums = np.bincount(groups, weights=difference)
        counts = np.bincount(groups)
        indices = rng.integers(0, len(counts), (resamples, len(counts)))
        bootstrap = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        reports[opponent] = dict(
            games=len(difference), unique_maps=len(counts),
            baseline_sha256=records[0]["checkpoint_sha256"],
            candidate_sha256=records[1]["checkpoint_sha256"],
            baseline_wld=[records[0][key] for key in ("wins", "losses", "draws")],
            candidate_wld=[records[1][key] for key in ("wins", "losses", "draws")],
            score_delta=float(difference.mean()),
            ci95=np.quantile(bootstrap, (0.025, 0.975)).tolist(),
            better=int((difference > 0).sum()), worse=int((difference < 0).sum()),
            same=int((difference == 0).sum()),
        )
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.resamples > 0
    result = dict(
        scope="Paired tuning validation; untouched final and hosted gates remain separate",
        seed=args.seed, map_cluster_resamples=args.resamples,
        contrasts={
            f"{baseline}_to_{candidate}": contrast(
                args.root, baseline, candidate, args.seed, args.resamples
            )
            for baseline, candidate in (("baseline", "control"), ("baseline", "fixed"), ("control", "fixed"))
        },
    )
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

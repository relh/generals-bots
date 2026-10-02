"""Compare paired held-out Classic population panels and optional move audits."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_arm(path):
    record = json.loads((path / "evaluation.json").read_text())
    arrays = {
        name: np.load(path / f"{name}.npy", allow_pickle=False)
        for name in ("initial_state_sha256", "initial_sides", "opponent_labels", "outcomes")
    }
    if "destination_audit" in record:
        arrays["destination_counts"] = np.load(path / "destination_counts.npy", allow_pickle=False)
    return record, arrays


def count_outcomes(values):
    return dict(wins=int((values == 1).sum()), losses=int((values == -1).sum()),
                draws=int((values == 0).sum()))


def cluster_interval(delta, hashes, *, seed, resamples):
    groups, labels = np.unique(hashes, return_inverse=True)
    sums = np.bincount(labels, weights=delta, minlength=len(groups))
    counts = np.bincount(labels, minlength=len(groups))
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        end = min(start + 256, resamples)
        selected = generator.integers(len(groups), size=(end - start, len(groups)))
        draws[start:end] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return len(groups), np.quantile(draws, (0.025, 0.975)).tolist()


def destination_summary(counts):
    if counts.ndim != 3 or counts.shape[1:] != (3, 4) or (counts < 0).any():
        raise ValueError("Destination audit must contain nonnegative game × phase × category counts")
    totals = counts.sum(axis=0, dtype=np.int64)
    actions = totals.sum(axis=1)
    return dict(counts=totals.tolist(), actions=actions.tolist(),
                fractions=np.divide(totals, actions[:, None],
                                    out=np.zeros((3, 4), np.float64),
                                    where=actions[:, None] != 0).tolist())


def compare(baseline, candidate, *, seed=35687, resamples=10000):
    if resamples <= 0:
        raise ValueError("Bootstrap resamples must be positive")
    base, base_arrays = load_arm(baseline)
    child, child_arrays = load_arm(candidate)
    shared = ("games", "pool_size", "seed", "sample_seed", "population_build_sha256",
              "frozen_policy_sha256", "opponent_weights", "coworld_classic_rules")
    if any(base[key] != child[key] for key in shared):
        raise ValueError("Population panels have different rules, maps, or opponents")
    if not base["coworld_classic_rules"] or base["games"] <= 0:
        raise ValueError("Expected nonempty official Coworld Classic panels")
    for key in ("initial_state_sha256", "initial_sides", "opponent_labels"):
        if not np.array_equal(base_arrays[key], child_arrays[key]):
            raise ValueError(f"Population panels differ in {key}")
    names = tuple(base["by_opponent_and_seat"])
    if names != tuple(child["by_opponent_and_seat"]):
        raise ValueError("Opponent names differ")
    hashes = base_arrays["initial_state_sha256"]
    sides = base_arrays["initial_sides"]
    labels = base_arrays["opponent_labels"]
    before = base_arrays["outcomes"]
    after = child_arrays["outcomes"]
    if any(len(array) != base["games"] for array in (hashes, sides, labels, before, after)):
        raise ValueError("Population array length differs from game count")
    if not (np.isin(before, (-1, 0, 1)).all() and np.isin(after, (-1, 0, 1)).all()):
        raise ValueError("Population outcomes must be signed wins, losses, or draws")
    if not np.isin(sides, (0, 1)).all() or not ((0 <= labels) & (labels < len(names))).all():
        raise ValueError("Population labels or seats are invalid")
    if count_outcomes(before) != {key: base[key] for key in ("wins", "losses", "draws")}:
        raise ValueError("Baseline outcome file disagrees with evaluation")
    if count_outcomes(after) != {key: child[key] for key in ("wins", "losses", "draws")}:
        raise ValueError("Candidate outcome file disagrees with evaluation")
    difference = after.astype(np.float64) - before.astype(np.float64)
    unique, interval = cluster_interval(difference, hashes, seed=seed, resamples=resamples)
    result = dict(scope="Paired first-episode official Classic population comparison",
                  games=base["games"], unique_initial_states=unique,
                  baseline_checkpoint_sha256=base["checkpoint_sha256"],
                  candidate_checkpoint_sha256=child["checkpoint_sha256"],
                  baseline=count_outcomes(before), candidate=count_outcomes(after),
                  paired_signed_score_delta=float(difference.mean()),
                  initial_state_cluster_ci95=interval,
                  bootstrap_seed=seed, bootstrap_resamples=resamples,
                  by_opponent_and_seat={})
    with_destinations = "destination_counts" in base_arrays and "destination_counts" in child_arrays
    if ("destination_counts" in base_arrays) != ("destination_counts" in child_arrays):
        raise ValueError("Only one arm recorded destination counts")
    if with_destinations:
        base_dest = base_arrays["destination_counts"]
        child_dest = child_arrays["destination_counts"]
        if base_dest.shape != (base["games"], 3, 4) or child_dest.shape != base_dest.shape:
            raise ValueError("Population destination arrays have different shapes")
        for key in ("phases", "categories"):
            if base["destination_audit"][key] != child["destination_audit"][key]:
                raise ValueError(f"Population destination {key} differ")
        for label, record, values in (("baseline", base, base_dest),
                                      ("candidate", child, child_dest)):
            if not np.array_equal(values.sum(axis=0), record["destination_audit"]["counts"]):
                raise ValueError(f"{label} destination counts disagree with evaluation")
        result["destination_audit"] = dict(
            phases=base["destination_audit"]["phases"],
            categories=base["destination_audit"]["categories"],
            baseline=destination_summary(base_dest), candidate=destination_summary(child_dest),
        )
    for index, name in enumerate(names):
        result["by_opponent_and_seat"][name] = {}
        for side in (0, 1):
            mask = (labels == index) & (sides == side)
            if not mask.any():
                raise ValueError(f"Opponent {name} lacks learner seat {side}")
            row = dict(games=int(mask.sum()), baseline=count_outcomes(before[mask]),
                       candidate=count_outcomes(after[mask]),
                       paired_signed_score_delta=float(difference[mask].mean()))
            for label, record, values in (("baseline", base, before), ("candidate", child, after)):
                reported = record["by_opponent_and_seat"][name][str(side)]
                if reported["games"] != row["games"] or any(
                    reported[key] != row[label][key] for key in ("wins", "losses", "draws")
                ):
                    raise ValueError(f"{label} opponent summary disagrees with outcome file")
            if with_destinations:
                row["destination_audit"] = dict(
                    baseline=destination_summary(base_dest[mask]),
                    candidate=destination_summary(child_dest[mask]),
                )
            result["by_opponent_and_seat"][name][str(side)] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=35687)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, seed=args.seed,
                     resamples=args.bootstrap_resamples)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "games", "unique_initial_states", "baseline", "candidate",
        "paired_signed_score_delta", "initial_state_cluster_ci95"
    )}), flush=True)


if __name__ == "__main__":
    main()

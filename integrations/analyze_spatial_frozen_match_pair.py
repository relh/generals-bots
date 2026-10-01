"""Compare two frozen Classic matches on the same maps and player seats.

Games drawn from one Coworld pool share maps. Resampling individual games
therefore overstates precision; the confidence interval resamples map groups.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(baseline: Path, candidate: Path, *, seed: int, resamples: int,
            allow_policy_mode_change: bool = False) -> dict:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    records = [json.loads((directory / "evaluation.json").read_text()) for directory in (baseline, candidate)]
    for record in records:
        if not record["held_out"] or record["smoke_cpu"]:
            raise ValueError("Both matches must be held-out GPU evaluations")
        if not record["coworld_classic_rules"]:
            raise ValueError("Both matches must use official Coworld Classic rules")
    settings = ("games", "seed", "pool_size", "opponent_sha256", "episode_limit",
                "coworld_classic_rules")
    # Historical records hard-coded "argmax" even when the frozen bundle
    # sampled actions. Compare the recorded contract when both records have it;
    # the exact opponent checkpoint still has to match in every case.
    if all("opponent_action_parameters" in record for record in records):
        settings += ("opponent_action_selection", "opponent_action_parameters")
    if not allow_policy_mode_change:
        settings += ("action_selection", "sample_seed", "sampling_temperature", "half_logit_bias")
    elif records[0]["checkpoint_sha256"] != records[1]["checkpoint_sha256"]:
        raise ValueError("Changing action selection requires the same checkpoint")
    for field in settings:
        if records[0][field] != records[1][field]:
            raise ValueError(f"Match settings differ: {field}")
    if not allow_policy_mode_change and records[0].get("split_sampling_temperature") != records[1].get("split_sampling_temperature"):
        raise ValueError("Match settings differ: split_sampling_temperature")
    if not allow_policy_mode_change and records[0].get("neutral_route_bias", 0.0) != records[1].get("neutral_route_bias", 0.0):
        raise ValueError("Match settings differ: neutral_route_bias")
    if not allow_policy_mode_change and records[0].get("owned_split_bias", 0.0) != records[1].get("owned_split_bias", 0.0):
        raise ValueError("Match settings differ: owned_split_bias")
    if not allow_policy_mode_change and records[0].get("safe_owned_split_bias", 0.0) != records[1].get("safe_owned_split_bias", 0.0):
        raise ValueError("Match settings differ: safe_owned_split_bias")
    if not allow_policy_mode_change and records[0].get("weak_owned_route_penalty", 0.0) != records[1].get("weak_owned_route_penalty", 0.0):
        raise ValueError("Match settings differ: weak_owned_route_penalty")
    if not allow_policy_mode_change and records[0].get("doomed_attack_route_penalty", 0.0) != records[1].get("doomed_attack_route_penalty", 0.0):
        raise ValueError("Match settings differ: doomed_attack_route_penalty")
    arrays = []
    for filename in ("initial_state_sha256.npy", "initial_sides.npy"):
        left, right = (np.load(directory / filename) for directory in (baseline, candidate))
        if not np.array_equal(left, right):
            raise ValueError(f"Matches are not paired: {filename}")
        arrays.append(left)
    hashes, sides = arrays
    scores = [np.load(directory / "outcomes.npy") for directory in (baseline, candidate)]
    for record, values in zip(records, scores, strict=True):
        if values.shape != (record["games"],) or hashes.shape != values.shape or sides.shape != values.shape:
            raise ValueError("Outcome or initial-state count differs from the match record")
        if not np.isin(values, (-1, 0, 1)).all():
            raise ValueError("Outcomes must be loss/draw/win values")
        if tuple(int((values == x).sum()) for x in (1, -1, 0)) != (
            record["wins"], record["losses"], record["draws"]
        ) or not np.isclose(values.mean(), record["score"]):
            raise ValueError("Saved outcomes disagree with the match record")
    if len(np.unique(sides)) != 2 or int((sides == 0).sum()) != int((sides == 1).sum()):
        raise ValueError("The paired match does not balance learner seats")

    difference = scores[1] - scores[0]
    _, groups = np.unique(hashes, return_inverse=True)
    sums = np.bincount(groups, weights=difference)
    counts = np.bincount(groups)
    rng = np.random.default_rng(seed)
    sampled = np.empty(resamples, np.float64)
    # Keep peak memory bounded even when the map pool or resample count grows.
    for start in range(0, resamples, 1024):
        stop = min(start + 1024, resamples)
        indices = rng.integers(0, len(sums), (stop - start, len(sums)))
        sampled[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    interval = np.quantile(sampled, (0.025, 0.975))
    verdict = "improvement" if interval[0] > 0 else "regression" if interval[1] < 0 else "inconclusive"
    return dict(
        baseline_sha256=records[0]["checkpoint_sha256"],
        candidate_sha256=records[1]["checkpoint_sha256"],
        opponent_sha256=records[0]["opponent_sha256"],
        match_seed=records[0]["seed"], pool_size=records[0]["pool_size"],
        sample_seed=records[1]["sample_seed"],
        coworld_classic_rules=records[0]["coworld_classic_rules"],
        initial_states_sha256=hashlib.sha256(hashes.tobytes()).hexdigest(),
        games=len(difference), unique_initial_maps=len(sums),
        baseline_wld=[records[0][field] for field in ("wins", "losses", "draws")],
        candidate_wld=[records[1][field] for field in ("wins", "losses", "draws")],
        score_delta=float(difference.mean()), ci95=interval.tolist(),
        better=int((difference > 0).sum()), worse=int((difference < 0).sum()),
        same=int((difference == 0).sum()), verdict=verdict,
        bootstrap_seed=seed, map_cluster_resamples=resamples,
        baseline_action_selection=records[0]["action_selection"],
        candidate_action_selection=records[1]["action_selection"],
        baseline_sampling_temperature=records[0]["sampling_temperature"],
        candidate_sampling_temperature=records[1]["sampling_temperature"],
        baseline_half_logit_bias=records[0].get("half_logit_bias", 0.0),
        candidate_half_logit_bias=records[1].get("half_logit_bias", 0.0),
        baseline_split_sampling_temperature=records[0].get("split_sampling_temperature"),
        candidate_split_sampling_temperature=records[1].get("split_sampling_temperature"),
        baseline_neutral_route_bias=records[0].get("neutral_route_bias", 0.0),
        candidate_neutral_route_bias=records[1].get("neutral_route_bias", 0.0),
        baseline_owned_split_bias=records[0].get("owned_split_bias", 0.0),
        candidate_owned_split_bias=records[1].get("owned_split_bias", 0.0),
        baseline_safe_owned_split_bias=records[0].get("safe_owned_split_bias", 0.0),
        candidate_safe_owned_split_bias=records[1].get("safe_owned_split_bias", 0.0),
        baseline_weak_owned_route_penalty=records[0].get("weak_owned_route_penalty", 0.0),
        candidate_weak_owned_route_penalty=records[1].get("weak_owned_route_penalty", 0.0),
        baseline_doomed_attack_route_penalty=records[0].get("doomed_attack_route_penalty", 0.0),
        candidate_doomed_attack_route_penalty=records[1].get("doomed_attack_route_penalty", 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--allow-policy-mode-change", action="store_true",
                        help="Compare greedy and sampled actions for the same checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.baseline, args.candidate, seed=args.seed, resamples=args.resamples,
                     allow_policy_mode_change=args.allow_policy_mode_change)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Independently audit the selected Puffer checkpoint and classic held-out score."""

import argparse
import hashlib
import json
import math
from pathlib import Path

VALIDATION_SEEDS = [901, 902, 903, 904, 905]
HELDOUT_SEEDS = [1001, 1002, 1003, 1004, 1005]
CHECKPOINTS = {"20": 20_480_000, "25": 25_600_000, "final": None}
TRAINED_STEPS = 31_457_280


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)


def audit_policy(
    workspace: Path,
    job_id: int,
    prefix: str,
    index: str,
    policy: dict,
    seeds: list[int],
    episodes: int,
    step: int | None,
) -> dict:
    run = workspace / f"gpu-batch-four-long-{job_id}-{index}"
    record, completed = read_json(run / "run.json"), read_json(run / "completed.json")
    require(completed["trained_timesteps"] == TRAINED_STEPS, f"policy {index}: incomplete training")
    require(record["config"]["seed"] not in VALIDATION_SEEDS + HELDOUT_SEEDS, f"policy {index}: seed reused")
    relative = policy["checkpoint_relative_path"]
    require(relative in completed["checkpoints"], f"policy {index}: checkpoint absent from training record")
    expected_relative = (
        f"checkpoints/metta_generals/{run.name}/{step:016d}.bin" if step is not None else completed["final_checkpoint"]
    )
    require(relative == expected_relative, f"policy {index}: wrong checkpoint step")
    require(policy["checkpoint_step"] == step, f"policy {index}: step metadata mismatch")
    checkpoint = run / relative
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    require(digest == policy["checkpoint_sha256"], f"policy {index}: checkpoint digest mismatch")

    raw_results = []
    for seed in seeds:
        evaluation = read_json(workspace / f"{prefix}-policy{index}-seed{seed}" / "evaluation.json")
        config = evaluation["config"]
        require(
            config["seeds"] == [seed] and config["episodes_per_seed"] == episodes, f"seed {seed}: eval config mismatch"
        )
        require(config["allow_environment_transfer"] is True, f"seed {seed}: environment transfer was not verified")
        require(evaluation["source_build"] == record["build"], f"seed {seed}: training build mismatch")
        require(evaluation["checkpoint_sha256"] == digest, f"seed {seed}: wrong checkpoint")
        build = evaluation["build"]["config"]
        environment = build["python_environment"]
        options, spec = environment["options"], environment["spec"]
        require(
            environment["factory"] == "integrations.metta_puffer:BatchedGeneralsPufferEnvironment",
            f"seed {seed}: wrong environment",
        )
        require(
            options["board_size"] == 10
            and options["horizon"] == 800
            and options["opponent"] == "mixed"
            and options["classic_maps"] is True
            and options["parallel_games"] == 16
            and spec["agents"] == 16,
            f"seed {seed}: wrong classic map or opponent distribution",
        )
        require(len(evaluation["results"]) == 1, f"seed {seed}: missing result")
        result = evaluation["results"][0]
        require(result["seed"] == seed and result["games"] >= episodes, f"seed {seed}: incomplete evaluation")
        require(0 <= result["perf"] <= 1, f"seed {seed}: invalid performance")
        raw_results.append(result)

    require(policy["results"] == raw_results, f"policy {index}: summary differs from raw results")
    games = sum(result["games"] for result in raw_results)
    performance = sum(result["games"] * result["perf"] for result in raw_results) / games
    require(policy["games"] == games and close(policy["perf"], performance), f"policy {index}: aggregate mismatch")
    return {
        "index": int(index),
        "checkpoint_step": step or TRAINED_STEPS,
        "checkpoint_sha256": digest,
        "batch_episodes": games,
        "individual_games": games * spec["agents"],
        "performance": performance,
        "seed_performance": {result["seed"]: result["perf"] for result in raw_results},
    }


def audit(workspace: Path, job_id: int) -> dict:
    candidates = []
    for label, step in CHECKPOINTS.items():
        prefix = f"gpu-batch-four-long-{job_id}-classic-v{label}"
        summary = read_json(workspace / f"{prefix}-summary.json")
        require(
            summary["training_job_id"] == job_id and summary["seeds"] == VALIDATION_SEEDS,
            f"{label}: wrong validation seeds",
        )
        require(set(summary["policies"]) == {"0", "1", "2", "3"}, f"{label}: missing policies")
        for index, policy in summary["policies"].items():
            candidates.append(audit_policy(workspace, job_id, prefix, index, policy, VALIDATION_SEEDS, 8, step))

    best = max(
        candidates,
        key=lambda item: (
            item["performance"],
            -item["index"],
            item["checkpoint_step"] if item["checkpoint_step"] != TRAINED_STEPS else 0,
        ),
    )
    prefix = f"gpu-batch-four-long-{job_id}-classic-heldout"
    summary = read_json(workspace / f"{prefix}-summary.json")
    require(summary["training_job_id"] == job_id and summary["seeds"] == HELDOUT_SEEDS, "wrong held-out seeds")
    require(set(summary["policies"]) == {str(best["index"])}, "held-out checkpoint was not selected on validation")
    policy = summary["policies"][str(best["index"])]
    step = None if best["checkpoint_step"] == TRAINED_STEPS else best["checkpoint_step"]
    heldout = audit_policy(workspace, job_id, prefix, str(best["index"]), policy, HELDOUT_SEEDS, 16, step)
    require(heldout["checkpoint_sha256"] == best["checkpoint_sha256"], "held-out checkpoint differs from validation")
    return {
        "training_job_id": job_id,
        "training_steps": TRAINED_STEPS,
        "validation_candidates": len(candidates),
        "selected_validation_performance": best["performance"],
        "heldout": heldout,
        "target": 0.60,
        "passed": heldout["performance"] >= 0.60,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("training_job_id", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    proof = audit(args.workspace, args.training_job_id)
    rendered = json.dumps(proof, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    if not proof["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

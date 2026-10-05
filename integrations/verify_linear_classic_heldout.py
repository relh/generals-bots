"""Audit the final linear Puffer policy and independent classic-map evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path

TRAINED_STEPS = 31_457_280
VALIDATION_SEEDS = [901, 902]
HELDOUT_SEEDS = [1001, 1002, 1003, 1004, 1005]


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-8)


def audit_evaluation(path: Path, record: dict, digest: str, seeds: list[int]) -> dict:
    evaluation = read(path / "evaluation.json")
    config = evaluation["config"]
    require(config["seeds"] == seeds and config["episodes_per_seed"] == 1, f"{path}: wrong evaluation seeds")
    require(config["allow_environment_transfer"] is True, f"{path}: environment transfer unverified")
    require(evaluation["source_build"] == record["build"], f"{path}: source build differs")
    require(evaluation["checkpoint_sha256"] == digest, f"{path}: checkpoint differs")
    build = evaluation["build"]["config"]
    require(build["fabric"]["platform"] == "cuda", f"{path}: model did not use CUDA")
    require(build["fabric"]["factory"] == "metta_training.fabric:linear_policy", f"{path}: wrong policy")
    environment = build["python_environment"]
    options, spec = environment["options"], environment["spec"]
    require(
        environment["factory"] == "integrations.metta_puffer:BatchedGeneralsPufferEnvironment"
        and options["board_size"] == 10
        and options["horizon"] == 800
        and options["opponent"] == "mixed"
        and options["classic_maps"] is True
        and options["parallel_games"] == spec["agents"] == 1024,
        f"{path}: wrong classic map, opponents, or parallel game count",
    )
    results = evaluation["results"]
    require([item["seed"] for item in results] == seeds, f"{path}: missing seed results")
    require(all(item["games"] >= 1 and 0 <= item["perf"] <= 1 for item in results), f"{path}: invalid results")
    games = sum(item["games"] for item in results)
    performance = sum(item["games"] * item["perf"] for item in results) / games
    return {
        "performance": performance,
        "batch_episodes": games,
        "individual_games": games * spec["agents"],
        "seed_performance": {item["seed"]: item["perf"] for item in results},
        "build": evaluation["build"],
    }


def audit(workspace: Path, training_job: int, validation_job: int, heldout_job: int) -> dict:
    candidates = []
    for index in range(2):
        run = workspace / f"linear-two-30m-{training_job}-{index}"
        record, completed = read(run / "run.json"), read(run / "completed.json")
        require(completed["trained_timesteps"] == TRAINED_STEPS, f"policy {index}: incomplete training")
        require(record["config"]["seed"] not in VALIDATION_SEEDS + HELDOUT_SEEDS, f"policy {index}: seed reused")
        require(record["config"]["total_timesteps"] == TRAINED_STEPS, f"policy {index}: wrong budget")
        relative = completed["final_checkpoint"]
        require(relative in completed["checkpoints"] and relative.endswith(f"{TRAINED_STEPS:016d}.bin"),
                f"policy {index}: wrong final checkpoint")
        digest = hashlib.sha256((run / relative).read_bytes()).hexdigest()
        suffix = "" if index == 0 else "-1"
        validation = audit_evaluation(
            workspace / f"linear-classic-validation-{validation_job}{suffix}",
            record, digest, VALIDATION_SEEDS,
        )
        candidates.append({"index": index, "checkpoint_sha256": digest, "run": run,
                           "record": record, "validation": validation})

    selected = max(candidates, key=lambda item: (item["validation"]["performance"], -item["index"]))
    selection = read(workspace / f"linear-classic-heldout-selection-{heldout_job}.json")
    require(selection["training_job_id"] == str(training_job)
            and selection["validation_job_id"] == str(validation_job), "selection references wrong jobs")
    require(selection["validation_seeds"] == VALIDATION_SEEDS, "selection used wrong seeds")
    require(selection["selected_index"] == selected["index"], "held-out policy was not selected on validation")
    for candidate, declared in zip(candidates, selection["candidates"], strict=True):
        require(declared["index"] == candidate["index"]
                and declared["checkpoint_sha256"] == candidate["checkpoint_sha256"]
                and close(declared["perf"], candidate["validation"]["performance"]),
                "selection summary differs from validation artifacts")
    heldout = audit_evaluation(
        workspace / f"linear-classic-heldout-{heldout_job}",
        selected["record"], selected["checkpoint_sha256"], HELDOUT_SEEDS,
    )
    require(heldout["build"] == selected["validation"]["build"], "held-out build changed")
    return {
        "training_job_id": training_job,
        "training_steps_per_policy": TRAINED_STEPS,
        "selected_index": selected["index"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "validation_performance": selected["validation"]["performance"],
        "heldout_performance": heldout["performance"],
        "heldout_seed_performance": heldout["seed_performance"],
        "heldout_individual_games": heldout["individual_games"],
        "target": 0.60,
        "passed": heldout["performance"] >= 0.60,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("training_job", type=int)
    parser.add_argument("validation_job", type=int)
    parser.add_argument("heldout_job", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    proof = audit(arguments.workspace, arguments.training_job, arguments.validation_job, arguments.heldout_job)
    rendered = json.dumps(proof, indent=2) + "\n"
    if arguments.output:
        arguments.output.write_text(rendered)
    print(rendered, end="")
    if not proof["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

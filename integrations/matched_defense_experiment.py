"""Run the current matched defense experiment after a measured geometry probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
from pathlib import Path

from integrations.classic_contract import validate_training_contract


def prepare_configs(build, run, *, agents, horizon, steps, seed):
    """Project one current experiment to a geometry, preserving its objective."""
    build, run = copy.deepcopy(build), copy.deepcopy(run)
    build["python_environment"]["options"]["parallel_games"] = agents
    build["python_environment"]["spec"]["agents"] = agents
    run["overrides"].update({"vec.total_agents": agents, "train.horizon": horizon})
    run.update(seed=seed, total_timesteps=steps)
    validate_training_contract(build, run)
    if steps % (agents * horizon) or steps < 2_600_000:
        raise ValueError("Matched budget must include complete rollouts beyond 2.6 million steps")
    if run["initialize"]["restore_learner"] is not False:
        raise ValueError("Both matched arms require fresh PPO optimizers")
    return build, run


def validate_qualification(directory, digest, build, run):
    """Bind the selected geometry to its retained completed GPU probe."""
    path = directory / "result.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("Qualification result hash differs")
    result = json.loads(path.read_text())
    audit = json.loads((directory / "pilot/training-audit.json").read_text())
    if result.get("qualified") is not True or result.get("audit") != audit:
        raise ValueError("Selected geometry did not complete qualification")
    overrides = run["overrides"]
    expected = {
        "environment_count": overrides["vec.total_agents"],
        "horizon": overrides["train.horizon"],
        "minibatch": overrides["train.minibatch_size"],
        "replay_ratio": overrides["train.replay_ratio"],
        "illegal_actions": 0,
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("Qualification geometry, optimizer batching or legality differs")
    if audit["steady_sps"] < 30_000 or audit["environment_steps"] < 2_600_000:
        raise ValueError("Qualification must complete the finite-gradient range at 30,000 SPS")
    if json.loads((directory / "build-config.json").read_text()) != build:
        raise ValueError("Qualification build, objective or population differs")
    probe_run = json.loads((directory / "pilot/config.json").read_text())
    # Checkpoint cadence and finite probe budget do not change rollout/optimizer algebra.
    probe_options = dict(probe_run["overrides"])
    target_options = dict(overrides)
    for options in (probe_options, target_options):
        options.pop("base.checkpoint_interval", None)
    if probe_options != target_options or probe_run["initialize"] != run["initialize"]:
        raise ValueError("Qualification learner or source initializer differs")
    return result


def verify_runtime(qualified_build, current_build):
    """A geometry measurement authorizes only the same compiled runtime sources."""
    before = json.loads((qualified_build / "build.json").read_text())
    after = json.loads((current_build / "build.json").read_text())
    for key in ("revision", "model_sha256", "environment_sha256", "config"):
        if before[key] != after[key]:
            raise ValueError("Qualified build identity differs: " + key)
    def native_sources(root):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "source/src").rglob("*")
            if path.is_file() and path.suffix in {".cu", ".cuh", ".h"}
        }
    before_sources = native_sources(qualified_build)
    if not before_sources or before_sources != native_sources(current_build):
        raise ValueError("Qualified native runtime source differs")


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--qualification-sha256", required=True)
    args = parser.parse_args()
    from integrations.cuda_runtime_binding import configure

    configure()
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle

    identity = verify_gpu_idle(visible_gpu_identity())
    os.environ["GENERALS_ALLOCATED_GPU_UUID"] = identity["uuid"]
    os.environ["CUDA_VISIBLE_DEVICES"] = identity["uuid"]
    build = json.loads((args.input / "build-config.json").read_text())
    run = json.loads((args.input / "config.json").read_text())
    validate_training_contract(build, run)
    qualification = validate_qualification(args.qualification, args.qualification_sha256, build, run)
    for split, seed, examples in (("train", 7700101, 2048), ("heldout", 7800101, 512)):
        manifest = json.loads((args.input / "defense" / split / "manifest.json").read_text())
        if manifest["root_seed"] != seed or manifest["examples"] != examples:
            raise ValueError("Matched intervention requires the sealed fresh public curriculum")
    from integrations.policy_trial import Trial

    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("qualification.json", "build", "control", "warm")):
        raise ValueError("Matched experiment output already contains an experiment")
    (args.output / "qualification.json").write_text(json.dumps({
        "result_sha256": args.qualification_sha256,
        "result": qualification,
    }, indent=2) + "\n")
    trial = Trial(args.input, args.output)
    trial.smoke()
    trial.build()
    verify_runtime(args.qualification / "build", args.output / "build")
    trial.train_arm("control")
    trial.distill()
    trial.train_arm("warm")
    trial.evaluate()
    (args.output / "COMPLETED.json").write_text(json.dumps({
        "steps_per_arm": run["total_timesteps"], "seed": run["seed"],
        "qualification_sha256": args.qualification_sha256,
        "scope": "Matched GPU learning and held-out population evaluation; no hosted promotion.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

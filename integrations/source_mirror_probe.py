"""Stage and run one bounded Classic PPO probe against a frozen source mirror.

The only population intervention is replacing the weakest historical frozen
opponent with the exact source actor and increasing that slot's sampling weight.
The learner, reward objective, sampler and other opponents remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
from pathlib import Path

from integrations.classic_contract import validate_training_contract

STEPS = 4_194_304
GAMES = 4_096
HORIZON = 128
MIRROR_WEIGHT = 12
SEED = 9_107_331
SOURCE_SHA256 = "f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14"
COLD_ASSET_SHA256 = "58925af1dbeaa46e17230d0ea856739232d4e057b28a14417e3d9ea65903a5b9"
TARGET_MODEL_SHA256 = "cead5dce2bc2f507363854d62bfa9ef9f2a76c7a231afabca6ba658fef6a1c61"
TARGET_ABI_SHA256 = "0c7a1fb0dfb646b394dd94fbabbad397182bfc3fe62fd739889f2cfd6a8de851"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "seal.json"
    }


def serializable_training_contract(build: dict, run: dict) -> dict:
    """Compare the effective Classic contract in its on-disk JSON form."""
    return json.loads(json.dumps(validate_training_contract(build, run)))


def prepare(source_input: Path, repository: Path, output: Path) -> dict:
    """Copy immutable launch inputs and record every changed configuration field."""
    if output.exists():
        raise ValueError("Probe input already exists; use a fresh directory")
    source_input, repository = source_input.resolve(strict=True), repository.resolve(strict=True)
    output.mkdir(parents=True)
    for name in ("assets/cold", "bundles/cold", "bundles/frozen", "curriculum", "puffer.git",
                 "raylib-5.5_linux_amd64"):
        shutil.copytree(source_input / name, output / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("._*", ".DS_Store"))
    for name in ("build-config.json", "config.json"):
        shutil.copy2(source_input / name, output / name)
    (output / "source").mkdir()
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", "HEAD"], stdout=subprocess.PIPE,
    )
    try:
        subprocess.run(["tar", "-x", "-C", str(output / "source")], stdin=archive.stdout, check=True)
    finally:
        assert archive.stdout is not None
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("Source Git archive failed")
    # Keep the original source bundle; replace precisely one frozen actor.
    old_slot = output / "bundles/frozen/0"
    shutil.rmtree(old_slot)
    shutil.copytree(output / "bundles/cold", old_slot)
    if digest(old_slot / "policy.bin") != SOURCE_SHA256:
        raise ValueError("Frozen mirror differs from selected source policy")
    asset = json.loads((output / "assets/cold/asset.json").read_text())
    if asset["factory_source_sha256"] != digest(output / "source/integrations/generals_fabric.py"):
        raise ValueError("Source model factory differs from cold asset")
    if (digest(output / "assets/cold/asset.json") != COLD_ASSET_SHA256
            or asset["model_sha256"] != TARGET_MODEL_SHA256
            or asset["abi_sha256"] != TARGET_ABI_SHA256):
        raise ValueError("Cold asset must use the proved clean H100 model identity")
    build_path, run_path = output / "build-config.json", output / "config.json"
    build, run = json.loads(build_path.read_text()), json.loads(run_path.read_text())
    original_build, original_run = json.loads(build_path.read_text()), json.loads(run_path.read_text())
    options = build["python_environment"]["options"]
    if options["opponent_weights"][0] != 1 or options["frozen_bundles"][0] != "/work/input/bundles/frozen/0":
        raise ValueError("Historical weak-opponent slot changed")
    if run["initialize"]["asset"] != "/work/input/assets/cold/asset.json" or run["initialize"]["restore_learner"]:
        raise ValueError("Expected cold source actor and fresh optimizer")
    if digest(output / "assets/cold/policy.bin") != SOURCE_SHA256:
        raise ValueError("Cold initializer differs from selected source policy")
    if run["initialize"]["manifest_sha256"] != digest(output / "assets/cold/asset.json"):
        raise ValueError("Cold initializer manifest differs")
    options["opponent_weights"][0] = MIRROR_WEIGHT
    options["parallel_games"] = GAMES
    build["python_environment"]["spec"]["agents"] = GAMES
    run["overrides"]["vec.total_agents"] = GAMES
    run["overrides"]["train.horizon"] = HORIZON
    run["total_timesteps"] = STEPS
    run["seed"] = SEED
    contract = serializable_training_contract(build, run)
    if contract["shaping_gamma"] != contract["learner_gamma"] or contract["engine_sha256"] == "":
        raise ValueError("Classic engine or discount contract differs")
    build_path.write_text(json.dumps(build, indent=2) + "\n")
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    diff = {
        "frozen_slot_0_policy_sha256": [digest(source_input / "bundles/frozen/0/policy.bin"), SOURCE_SHA256],
        "opponent_weights_0": [original_build["python_environment"]["options"]["opponent_weights"][0], MIRROR_WEIGHT],
        "parallel_games": [original_build["python_environment"]["options"]["parallel_games"], GAMES],
        "rollout_horizon": [original_run["overrides"]["train.horizon"], HORIZON],
        "agent_steps": [original_run["total_timesteps"], STEPS],
        "seed": [original_run["seed"], SEED],
    }
    proof = {
        "experiment": "source-mirror-ppo-probe-v1",
        "source_revision": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip(),
        "source_policy_sha256": SOURCE_SHA256,
        "cold_asset_sha256": digest(output / "assets/cold/asset.json"),
        "changes": diff,
        "contract": contract,
        "population_weights": options["opponent_weights"],
        "sampler": asset["sampler"],
        "scope": "One bounded GPU throughput and PPO probe; no hosted replay input or promotion",
    }
    (output / "probe-intent.json").write_text(json.dumps(proof, indent=2) + "\n")
    # BuildKit may read the context as a different UID. Preserve executable
    # bits while making every input and parent directory traversable.
    for path in (output, *output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Portable probe input must not contain symlinks")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | (0o555 if path.is_dir() else 0o444))
    seal = file_hashes(output)
    (output / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    (output / "seal.json").chmod(0o644)
    return proof


def run_probe(inputs: Path, output: Path) -> None:
    """Execute one GPU probe; the audit rejects incomplete or slow training."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    inputs = inputs.resolve(strict=True)
    if file_hashes(inputs) != json.loads((inputs / "seal.json").read_text()):
        raise ValueError("Staged probe input differs from its file seal")
    proof = json.loads((inputs / "probe-intent.json").read_text())
    if proof["source_policy_sha256"] != SOURCE_SHA256:
        raise ValueError("Source policy lineage differs")
    asset_path = inputs / "assets/cold/asset.json"
    asset = json.loads(asset_path.read_text())
    if (digest(asset_path) != COLD_ASSET_SHA256 or proof["cold_asset_sha256"] != COLD_ASSET_SHA256
            or asset["model_sha256"] != TARGET_MODEL_SHA256
            or asset["abi_sha256"] != TARGET_ABI_SHA256):
        raise ValueError("Cold asset must match the proved clean H100 target")
    build, run = json.loads((inputs / "build-config.json").read_text()), json.loads((inputs / "config.json").read_text())
    if serializable_training_contract(build, run) != proof["contract"]:
        raise ValueError("Effective Classic training contract differs")
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
    from integrations.cuda_runtime_binding import configure

    configure()
    gpu = verify_gpu_idle(visible_gpu_identity())
    os.environ["GENERALS_ALLOCATED_GPU_UUID"] = gpu["uuid"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    from integrations.classic_position_curriculum import configure_positions
    from integrations.policy_execution import training_audit
    from integrations.policy_trial import Trial

    configure_positions(build["python_environment"]["options"], inputs / "curriculum/manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "gpu-preflight.json").write_text(json.dumps(gpu, indent=2) + "\n")
    (output / "build-config.json").write_text(json.dumps(build, indent=2) + "\n")
    probe = output / "probe"
    probe.mkdir()
    (probe / "config.json").write_text(json.dumps(run, indent=2) + "\n")
    trial = Trial(inputs, output)
    trial.build()
    # The selected source actor is the control bundle in Trial; the other
    # arm denotes the separate distilled candidate and has no bundle here.
    trial.sampling_gate("control")
    shutil.copy2(output / "control/sampling-gate.json", probe / "sampling-gate.json")
    trial.call("launch_spatial_selfplay_training", [
        "preflight", "--build", output / "build", "--config", probe / "config.json",
        "--output", probe / "run",
    ], name="preflight", seconds=300, arm="probe")
    trial.call("launch_spatial_selfplay_training", [
        "train", "--build", output / "build", "--config", probe / "config.json",
        "--output", probe / "run",
    ], name="train", seconds=720, arm="probe", training=True)
    audit = training_audit(probe, run)
    if audit["environment_steps"] != STEPS or audit["steady_sps"] < 30_000:
        raise ValueError("Source-mirror probe did not qualify")
    (output / "QUALIFIED.json").write_text(json.dumps({
        "intent": proof, "gpu": gpu, "audit": audit,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare")
    stage.add_argument("--source-input", type=Path, required=True)
    stage.add_argument("--repository", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.source_input, args.repository, args.output), indent=2))
    else:
        run_probe(args.input, args.output)


if __name__ == "__main__":
    main()

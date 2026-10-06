"""Stage and run one bounded H100 Product-policy throughput qualification."""

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
SEED = 10_943_317
SOURCE_ASSET_SHA256 = "8a2a324c922bd76dcd4efa618ecceca2cdf88c2ea2600cf5551c891b1fee13a9"
SOURCE_POLICY_SHA256 = "9c55dc3b167746f0153f5de50afaa6cba9cfb4a8858107d524c05d988c9147e9"
TRANSPLANT_PROOF_SHA256 = "f9c698cad0a421d531e3be43df2be31534da39032a354e628a35a1a949c98b90"
PARITY_PROOF_SHA256 = "d914d3fa870190867446baa148206293ff5b1b6e7b6aeb40106c1aec37124025"
CONTROL_WEIGHTS = [1, 2, 1, 1, 1, 4, 5, 5, 11, 17, 3, 4, 16]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(root):
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "seal.json"}


def prepare(source_input: Path, repository: Path, target_asset: Path, target_bundle: Path,
            pool: Path, transplant_proof: Path, parity_proof: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    source_input, repository = source_input.resolve(strict=True), repository.resolve(strict=True)
    if (digest(target_asset) != SOURCE_ASSET_SHA256
            or digest(transplant_proof) != TRANSPLANT_PROOF_SHA256
            or digest(parity_proof) != PARITY_PROOF_SHA256):
        raise ValueError("Product source or parity receipts differ")
    source_metadata = json.loads(target_asset.read_text())
    if source_metadata["policy_sha256"] != SOURCE_POLICY_SHA256:
        raise ValueError("Product policy source differs")
    pool_receipt = pool / "migration-receipt.json"
    migrated = json.loads(pool_receipt.read_text())
    if (migrated["schema"] != "generals-product-frozen-pool-migration-v1"
            or len(migrated["frozen_opponents"]) != 10
            or migrated["frozen_opponents"][0]["new_policy_sha256"] != SOURCE_POLICY_SHA256):
        raise ValueError("Frozen opponent migration receipt differs")
    output.mkdir(parents=True)
    for name in ("curriculum", "puffer.git", "raylib-5.5_linux_amd64"):
        shutil.copytree(source_input / name, output / name, ignore=shutil.ignore_patterns("._*", ".DS_Store"))
    for name in ("build-config.json", "config.json"):
        shutil.copy2(source_input / name, output / name)
    shutil.copytree(target_asset.parent, output / "assets/cold",
                    ignore=shutil.ignore_patterns("policy.bin.learner"))
    shutil.copytree(target_bundle, output / "bundles/cold")
    shutil.copytree(pool / "bundles/frozen", output / "bundles/frozen")
    shutil.copy2(pool_receipt, output / "pool-migration-receipt.json")
    shutil.copy2(transplant_proof, output / "transplant-proof.json")
    shutil.copy2(parity_proof, output / "parity-proof.json")
    (output / "source").mkdir()
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", "HEAD"], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-x", "-C", str(output / "source")], stdin=archive.stdout, check=True)
    finally:
        assert archive.stdout is not None
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("Product source archive failed")
    factory_sha = digest(output / "source/integrations/generals_fabric.py")
    if source_metadata["factory_source_sha256"] != factory_sha:
        raise ValueError("Product factory source differs from initialized asset")
    asset_path = output / "assets/cold/asset.json"
    if digest(asset_path) != SOURCE_ASSET_SHA256:
        raise ValueError("Staged Product initializer differs")
    for row in migrated["frozen_opponents"]:
        path = output / f"bundles/frozen/{row['slot']}"
        if (digest(path / "asset.json") != row["new_asset_sha256"]
                or digest(path / "spatial-policy.json") != row["new_bundle_sha256"]
                or digest(path / "policy.bin") != row["new_policy_sha256"]):
            raise ValueError("Staged frozen opponent differs")
    build_path, run_path = output / "build-config.json", output / "config.json"
    build, run = json.loads(build_path.read_text()), json.loads(run_path.read_text())
    options = build["python_environment"]["options"]
    if (options["opponent_weights"] != [12, *CONTROL_WEIGHTS[1:]]
            or len(options["frozen_bundles"]) != 10
            or options["scripted_opponents"] != ["expander_harvester", "sentinel", "classic_siege_padded"]):
        raise ValueError("Original 13-opponent pool differs")
    options["opponent_weights"] = CONTROL_WEIGHTS
    options["parallel_games"] = GAMES
    build["python_environment"]["spec"]["agents"] = GAMES
    if build["fabric"] != source_metadata["fabric"]:
        raise ValueError("Build model differs from Product initializer")
    run["initialize"] = {"asset": "/work/input/assets/cold/asset.json",
                         "manifest_sha256": SOURCE_ASSET_SHA256, "restore_learner": False}
    run["overrides"].update({"vec.total_agents": GAMES, "train.horizon": HORIZON,
                             "train.minibatch_size": 8192, "train.replay_ratio": .5,
                             "base.checkpoint_interval": 8})
    run["total_timesteps"] = STEPS
    run["seed"] = SEED
    contract = json.loads(json.dumps(validate_training_contract(build, run)))
    if (contract["engine_sha256"] != "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"
            or contract["shaping_gamma"] != contract["learner_gamma"]
            or not contract["map_options"]["coworld_classic_rules"]):
        raise ValueError("Official Classic engine or discount contract differs")
    build_path.write_text(json.dumps(build, indent=2) + "\n")
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    proof = {"schema": "generals-source-global-product-throughput-intent-v1",
             "source_revision": subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"],
                                                       text=True).strip(),
             "source_asset_sha256": SOURCE_ASSET_SHA256,
             "source_policy_sha256": SOURCE_POLICY_SHA256,
             "source_bundle_sha256": digest(output / "bundles/cold/spatial-policy.json"),
             "transplant_proof_sha256": TRANSPLANT_PROOF_SHA256,
             "parity_proof_sha256": PARITY_PROOF_SHA256,
             "pool_migration_receipt_sha256": digest(output / "pool-migration-receipt.json"),
             "pool_old_to_new": migrated["frozen_opponents"],
             "contract": contract,
             "seed": SEED, "steps": STEPS, "opponent_weights": CONTROL_WEIGHTS,
             "sampler": source_metadata["sampler"],
             "scope": "One H100 throughput probe; no matched long PPO or policy promotion"}
    (output / "probe-intent.json").write_text(json.dumps(proof, indent=2) + "\n")
    for path in (output, *output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Rootless context input cannot contain symlinks")
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | (0o555 if path.is_dir() else 0o444))
    (output / "seal.json").write_text(json.dumps(file_hashes(output), indent=2) + "\n")
    (output / "seal.json").chmod(0o644)
    return proof


def run_probe(inputs: Path, output: Path):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    inputs = inputs.resolve(strict=True)
    if file_hashes(inputs) != json.loads((inputs / "seal.json").read_text()):
        raise ValueError("Sealed Product input differs")
    intent = json.loads((inputs / "probe-intent.json").read_text())
    if (intent["source_asset_sha256"] != SOURCE_ASSET_SHA256
            or intent["transplant_proof_sha256"] != digest(inputs / "transplant-proof.json")
            or intent["parity_proof_sha256"] != digest(inputs / "parity-proof.json")
            or intent["pool_migration_receipt_sha256"] != digest(inputs / "pool-migration-receipt.json")):
        raise ValueError("Product proof chain differs")
    build = json.loads((inputs / "build-config.json").read_text())
    run = json.loads((inputs / "config.json").read_text())
    if json.loads(json.dumps(validate_training_contract(build, run))) != intent["contract"]:
        raise ValueError("Effective Classic contract differs")
    from integrations.cuda_runtime_binding import configure
    from integrations.slurm_s3_job import verify_gpu_idle, visible_gpu_identity

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
    trial.call("launch_spatial_selfplay_training", [
        "build", "--config", output / "build-config.json", "--output", output / "build",
    ], name="build", seconds=420)
    control = output / "control"
    control.mkdir()
    trial.sampling_gate("control")
    shutil.copy2(control / "sampling-gate.json", probe / "sampling-gate.json")
    trial.call("launch_spatial_selfplay_training", [
        "preflight", "--build", output / "build", "--config", probe / "config.json",
        "--output", probe / "run",
    ], name="preflight", seconds=240, arm="probe")
    trial.call("launch_spatial_selfplay_training", [
        "train", "--build", output / "build", "--config", probe / "config.json",
        "--output", probe / "run",
    ], name="train", seconds=600, arm="probe", training=True)
    audit = training_audit(probe, run)
    if (audit["environment_steps"] != STEPS or audit["steady_sps"] < 30_000
            or len(audit["opponent_counts_by_seat"]) != 13):
        raise ValueError("Product architecture throughput or integrity gate failed")
    (output / "QUALIFIED.json").write_text(json.dumps({"intent": intent, "gpu": gpu, "audit": audit,
                                                      "source_gate_sha256": digest(control / "sampling-gate.json")},
                                                     indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare")
    for name in ("source-input", "repository", "target-asset", "target-bundle", "pool",
                 "transplant-proof", "parity-proof", "output"):
        stage.add_argument("--" + name, type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.source_input, args.repository, args.target_asset, args.target_bundle,
                                 args.pool, args.transplant_proof, args.parity_proof, args.output), indent=2))
    else:
        run_probe(args.input, args.output)


if __name__ == "__main__":
    main()

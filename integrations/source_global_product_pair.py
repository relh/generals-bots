"""Matched Product PPO: train Q versus an exact zero-Q functional control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import subprocess
from pathlib import Path

import numpy as np

from integrations.classic_contract import validate_training_contract
from integrations.source_global_product_probe import file_hashes

ARMS = ("control", "product")
PLAN_FILE = "integrations/source_global_product_pair_plan.json"


def phase(name: str, event: str, **details) -> None:
    print("PAIR_PHASE " + json.dumps({"name": name, "event": event, **details}, sort_keys=True), flush=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def qualified_probe(marker: Path, terminal: Path, activation_report: Path,
                    marker_sha256: str, plan: dict, probe_intent: dict) -> dict:
    pinned = ("required_probe_job_id", "required_probe_marker_sha256", "required_probe_activation_sha256",
              "source_product_asset_sha256", "pool_migration_receipt_sha256", "abi_proof_sha256",
              "source_parity_proof_sha256", "required_probe_native_source_parity_sha256",
              "required_probe_original_terminal_sha256", "required_probe_original_terminal_audit_sha256",
              "required_probe_original_artifact_sha256", "required_probe_original_failure_log_sha256",
              "required_probe_corrected_activation_auditor_sha256")
    if any(not isinstance(plan.get(key), str) or not plan[key] for key in pinned):
        raise ValueError("Repaired Product probe identities remain unpinned")
    if digest(marker) != marker_sha256 or marker_sha256 != plan["required_probe_marker_sha256"]:
        raise ValueError("Product probe marker hash differs")
    result = json.loads(marker.read_text())
    receipt = json.loads(terminal.read_text())
    activation = json.loads(activation_report.read_text())
    audit = result["audit"]
    movement = activation["q_u_v_movement"]
    effect = activation.get("legal_logit_delta_max_abs")
    if type(effect) not in (int, float) or not math.isfinite(effect):
        raise ValueError("Product activation effect is not finite")
    if (result.get("schema") != "generals-product-logical-muon-offline-qualified-v1"
            or receipt.get("job_id") != plan["required_probe_job_id"]
            or receipt.get("status") != "failed"
            or receipt.get("exit_code") != 1
            or result.get("original_job_status") != "failed"
            or result.get("original_exit_code") != 1
            or result.get("recovery_reason") != "raw_sampler_dictionary_defaults_only"
            or digest(terminal) != plan["required_probe_original_terminal_sha256"]
            or any(result.get(key) != plan["required_probe_" + key] for key in (
                "original_terminal_sha256", "original_terminal_audit_sha256", "original_artifact_sha256",
                "original_failure_log_sha256", "corrected_activation_auditor_sha256"))
            or result["terminal_audit"]["status"] != "failed"
            or result["terminal_audit"]["job_id"] != receipt.get("job_id")
            or result["terminal_audit"]["artifact_sha256"] != result["original_artifact_sha256"]
            or result["terminal_audit"]["training"] != audit
            or receipt.get("restarts_used") != 0
            or result["intent"] != probe_intent
            or probe_intent.get("schema") != "generals-product-logical-muon-probe-intent-v1"
            or result.get("source_parity_sha256") != plan["required_probe_native_source_parity_sha256"]
            or result.get("abi_proof_sha256") != plan["abi_proof_sha256"]
            or result.get("migration_receipt_sha256") != plan["pool_migration_receipt_sha256"]
            or result.get("initial_policy_sha256") != plan["source_product_policy_sha256"]
            or result.get("activation_gate_sha256") != plan["required_probe_activation_sha256"]
            or digest(activation_report) != plan["required_probe_activation_sha256"]
            or result.get("activation_gate") != activation
            or activation.get("passed") is not True
            or activation.get("source_policy_sha256") != plan["source_product_policy_sha256"]
            or activation.get("trained_checkpoint_sha256") != result.get("checkpoint_sha256")
            or activation.get("trained_bundle_manifest_sha256") != result.get("bundle_manifest_sha256")
            or activation.get("native_serving_parity_sha256") != result.get("trained_parity_sha256")
            or activation.get("views_sha256") != probe_intent.get("activation_views_sha256")
            or effect <= 1e-3
            or activation.get("min_legal_logit_delta_max_abs") != 1e-3
            or set(movement) != {"product_local_kernel", "product_global_kernel", "product_action_kernel"}
            or any(item.get("changed_words", 0) <= 0 for item in movement.values())
            or audit["environment_steps"] != plan["required_probe_steps"]
            or audit["steady_sps"] < plan["training"]["steady_sps_floor"]
            or audit["illegal_actions"] != 0
            or any(audit["reward_audit"][key] for key in (
                "nonfinite_rewards", "native_clipped_rewards", "native_clipped_terminal_rewards"))
            or len(audit["opponent_counts_by_seat"]) != 13
            or not all(counts["0"] == counts["1"] and counts["0"] > 0
                       for counts in audit["opponent_counts_by_seat"].values())):
        raise ValueError("Product probe does not qualify the exact source and population")
    return result


def verify_source_proofs(inputs: Path, plan: dict, intent: dict) -> None:
    transplant = json.loads((inputs / "transplant-proof.json").read_text())
    parity = json.loads((inputs / "parity-proof.json").read_text())
    if (digest(inputs / "transplant-proof.json") != intent["transplant_proof_sha256"]
            or digest(inputs / "parity-proof.json") != intent["parity_proof_sha256"]
            or transplant["source_policy_sha256"] != plan["source_old_policy_sha256"]
            or transplant["target_policy_sha256"] != plan["source_product_policy_sha256"]
            or transplant["bitwise_equal_source_logits"] is not True
            or transplant["zero_head"] is not True
            or parity["policy_sha256"] != plan["source_product_policy_sha256"]
            or parity["source_direct_portable_max_abs"] > 2e-5
            or parity["nonzero_direct_portable_max_abs"] > 2e-5):
        raise ValueError("Product source transplant or serving proof differs")


def prepare(probe_input: Path, repository: Path, marker: Path, terminal: Path,
            activation_report: Path, marker_sha256: str, output: Path) -> None:
    """Stage after reviewing the terminal Product probe; never alter its input."""
    if output.exists():
        raise FileExistsError(output)
    probe_input, repository = probe_input.resolve(strict=True), repository.resolve(strict=True)
    if file_hashes(probe_input) != json.loads((probe_input / "seal.json").read_text()):
        raise ValueError("Product probe input differs from its seal")
    plan = json.loads((repository / PLAN_FILE).read_text())
    intent = json.loads((probe_input / "probe-intent.json").read_text())
    qualified_probe(marker, terminal, activation_report, marker_sha256, plan, intent)
    if (intent["source_policy_sha256"] != plan["source_product_policy_sha256"]
            or intent["source_asset_sha256"] != plan["source_product_asset_sha256"]
            or intent["pool_migration_receipt_sha256"] != plan["pool_migration_receipt_sha256"]
            or intent["abi_proof_sha256"] != plan["abi_proof_sha256"]
            or intent["parity_proof_sha256"] != plan["source_parity_proof_sha256"]
            or intent["opponent_weights"] != plan["opponent_weights"]):
        raise ValueError("Product probe source or migrated population differs from preregistration")
    verify_source_proofs(probe_input, plan, intent)
    if digest(repository / "integrations/audit_product_activation.py") != plan[
            "required_probe_corrected_activation_auditor_sha256"]:
        raise ValueError("Matched pair must use the independently re-audited activation repair")
    output.mkdir(parents=True)
    for name in ("assets", "bundles", "curriculum", "puffer.git", "raylib-5.5_linux_amd64", "leader-root"):
        shutil.copytree(probe_input / name, output / name, ignore=shutil.ignore_patterns("._*", ".DS_Store"))
    for name in ("build-config.json", "config.json", "probe-intent.json",
                 "pool-migration-receipt.json", "transplant-proof.json", "parity-proof.json", "abi-proof.json",
                 "activation-views.npz", "activation-views.json"):
        shutil.copy2(probe_input / name, output / name)
    (output / "source").mkdir()
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", "HEAD"], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-x", "-C", str(output / "source")], stdin=archive.stdout, check=True)
    finally:
        assert archive.stdout is not None
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("Pair source archive failed")
    if digest(output / "source/integrations/generals_fabric.py") != json.loads(
            (output / "assets/cold/asset.json").read_text())["factory_source_sha256"]:
        raise ValueError("Pair source factory differs from Product asset")
    shutil.copy2(repository / PLAN_FILE, output / "plan.json")
    shutil.copy2(marker, output / "qualified-probe-marker.json")
    shutil.copy2(terminal, output / "qualified-probe-terminal.json")
    shutil.copy2(activation_report, output / "qualified-probe-activation.json")
    build = json.loads((output / "build-config.json").read_text())
    run = json.loads((output / "config.json").read_text())
    train = plan["training"]
    if (build["python_environment"]["options"]["opponent_weights"] != plan["opponent_weights"]
            or run["total_timesteps"] != plan["required_probe_steps"]
            or run["seed"] != intent["seed"]):
        raise ValueError("Probe training input differs before matched extension")
    run["total_timesteps"] = train["steps_per_arm"]
    run["seed"] = train["seed"]
    if (run["overrides"]["vec.total_agents"] != train["parallel_games"]
            or run["overrides"]["train.horizon"] != train["horizon"]
            or run["overrides"]["train.minibatch_size"] != train["minibatch"]
            or run["overrides"]["train.replay_ratio"] != train["replay_ratio"]):
        raise ValueError("Matched PPO geometry differs from qualified Product probe")
    (output / "config.json").chmod(0o600)
    (output / "config.json").write_text(json.dumps(run, indent=2) + "\n")
    contract = json.loads(json.dumps(validate_training_contract(build, run)))
    if contract != intent["contract"]:
        raise ValueError("Classic source, reward, curriculum or discount contract differs")
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    (output / "pair-lineage.json").write_text(json.dumps({
        "source_revision": revision,
        "probe_marker_sha256": marker_sha256,
        "probe_terminal_sha256": digest(terminal),
        "probe_activation_sha256": digest(activation_report),
        "probe_input_seal_sha256": digest(probe_input / "seal.json"),
        "plan_sha256": digest(output / "plan.json"),
        "source_product_policy_sha256": plan["source_product_policy_sha256"],
    }, indent=2) + "\n")
    for path in (output, *output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Pair context cannot contain symlinks")
        path.chmod(stat.S_IMODE(path.stat().st_mode) | (0o555 if path.is_dir() else 0o444))
    (output / "seal.json").write_text(json.dumps(file_hashes(output), indent=2) + "\n")
    (output / "seal.json").chmod(0o644)


def product_tensors(bundle: Path) -> dict[str, np.ndarray]:
    with np.load(bundle / "weights.npz", allow_pickle=False) as weights:
        return {name: weights[name].copy() for name in (
            "product_local_kernel", "product_global_kernel", "product_action_kernel")}


def verify_control(source_bundle: Path, control_bundle: Path) -> None:
    old, control = product_tensors(source_bundle), product_tensors(control_bundle)
    for name in ("product_local_kernel", "product_global_kernel"):
        if not np.array_equal(old[name], control[name]):
            raise ValueError("Zero-Q control changed its untrained Product projection: " + name)
    q = control["product_action_kernel"]
    if q.shape != (8, 8) or not np.array_equal(q, np.zeros((8, 8), np.float32)):
        raise ValueError("Control Q is not bitwise zero after native PPO")


def verify_serving_parity(report: Path, checkpoint: Path, bundle: Path, source: Path,
                          intent: dict) -> None:
    parity = json.loads(report.read_text())
    tolerances = {"max_logit_difference": 2e-5, "max_action_probability_difference": 1e-5,
                  "max_rollout_transform_difference": 1e-5}
    if (parity.get("checkpoint_sha256") != digest(checkpoint)
            or parity.get("bundle_manifest_sha256") != digest(bundle / "spatial-policy.json")
            or parity.get("factory_source_sha256") != digest(source / "integrations/generals_fabric.py")
            or parity.get("engine_sha256") != intent["contract"]["engine_sha256"]
            or parity.get("public_states", 0) < 16
            or parity.get("matching_top_actions") != parity.get("public_states")
            or any(type(parity.get(key)) not in (int, float) or not math.isfinite(parity[key])
                   or parity[key] > limit for key, limit in tolerances.items())):
        raise ValueError("Matched checkpoint native-to-serving parity differs")


def run_pair(inputs: Path, output: Path) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    phase("qualification", "start")
    inputs = inputs.resolve(strict=True)
    if file_hashes(inputs) != json.loads((inputs / "seal.json").read_text()):
        raise ValueError("Matched Product input differs from its seal")
    source = inputs / "source"
    if digest(inputs / "plan.json") != digest(source / PLAN_FILE):
        raise ValueError("Preregistered Product pair plan differs from source")
    plan = json.loads((inputs / "plan.json").read_text())
    lineage = json.loads((inputs / "pair-lineage.json").read_text())
    intent = json.loads((inputs / "probe-intent.json").read_text())
    qualified = qualified_probe(inputs / "qualified-probe-marker.json", inputs / "qualified-probe-terminal.json",
                                inputs / "qualified-probe-activation.json", lineage["probe_marker_sha256"], plan,
                                intent)
    verify_source_proofs(inputs, plan, intent)
    if digest(source / "integrations/audit_product_activation.py") != plan[
            "required_probe_corrected_activation_auditor_sha256"]:
        raise ValueError("Matched pair activation auditor differs from recovered qualification")
    if (lineage["source_product_policy_sha256"] != plan["source_product_policy_sha256"]
            or digest(inputs / "assets/cold/policy.bin") != plan["source_product_policy_sha256"]
            or digest(inputs / "assets/cold/asset.json") != plan["source_product_asset_sha256"]
            or digest(inputs / "pool-migration-receipt.json") != plan["pool_migration_receipt_sha256"]
            or digest(inputs / "abi-proof.json") != plan["abi_proof_sha256"]
            or digest(inputs / "activation-views.npz") != intent["activation_views_sha256"]
            or lineage["probe_activation_sha256"] != plan["required_probe_activation_sha256"]):
        raise ValueError("Product source or frozen opponent migration differs")
    build = json.loads((inputs / "build-config.json").read_text())
    config = json.loads((inputs / "config.json").read_text())
    train, dev = plan["training"], plan["development"]
    if (config["seed"] != train["seed"] or config["total_timesteps"] != train["steps_per_arm"]
            or build["python_environment"]["options"]["opponent_weights"] != plan["opponent_weights"]
            or config["overrides"]["vec.total_agents"] != train["parallel_games"]
            or config["overrides"]["train.horizon"] != train["horizon"]
            or config["overrides"]["train.minibatch_size"] != train["minibatch"]
            or config["overrides"]["train.replay_ratio"] != train["replay_ratio"]
            or json.loads(json.dumps(validate_training_contract(build, config))) != intent["contract"]):
        raise ValueError("Matched PPO configuration differs from preregistration")
    from integrations.cuda_runtime_binding import configure
    configure()
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
    gpu = verify_gpu_idle(visible_gpu_identity())
    os.environ["GENERALS_ALLOCATED_GPU_UUID"] = gpu["uuid"]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    from integrations.classic_position_curriculum import configure_positions
    from integrations.policy_execution import training_audit
    from integrations.policy_trial import Trial
    configure_positions(build["python_environment"]["options"], inputs / "curriculum/manifest.json")
    output.mkdir(parents=True, exist_ok=False)
    (output / "gpu-preflight.json").write_text(json.dumps(gpu, indent=2) + "\n")
    source_policy = plan["source_product_policy_sha256"]
    source_tensors = product_tensors(inputs / "bundles/cold")
    if not np.array_equal(source_tensors["product_action_kernel"], np.zeros((8, 8), np.float32)):
        raise ValueError("Product source Q must start bitwise zero")
    phase("qualification", "end", probe_sps=qualified["audit"]["steady_sps"])
    (output / "build-config.json").write_text(json.dumps(build, indent=2) + "\n")
    shared = Trial(inputs, output)
    if shared.validate_source_initializer(config)["fabric"] != build["fabric"]:
        raise ValueError("Matched Product model differs from its source initializer")
    phase("native_build", "start")
    shared.build()
    phase("native_build", "end")
    phase("sampling_gate", "start")
    shared.sampling_gate("control")
    phase("sampling_gate", "end")
    source_gate = output / "control/sampling-gate.json"
    arm_receipts = {}
    for arm in ARMS:
        target = output / arm
        target.mkdir(exist_ok=arm == "control")
        (target / "build-config.json").write_text(json.dumps(build, indent=2) + "\n")
        probe = target / "probe"
        probe.mkdir()
        (probe / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        trial = Trial(inputs, target)
        shutil.copy2(source_gate, probe / "sampling-gate.json")
        phase("preflight", "start", arm=arm)
        trial.call("launch_spatial_selfplay_training", ["preflight", "--build", output / "build",
             "--config", probe / "config.json", "--output", probe / "run"],
             name="preflight", seconds=300, arm="probe")
        phase("preflight", "end", arm=arm)
        phase("ppo", "start", arm=arm, steps=train["steps_per_arm"])
        trial.call("launch_spatial_selfplay_training", ["train", "--build", output / "build",
             "--config", probe / "config.json", "--output", probe / "run"],
             name="train", seconds=1800, arm="probe", training=True,
             product_head_frozen=arm == "control")
        audit = training_audit(probe, config)
        if audit["steady_sps"] < train["steady_sps_floor"]:
            raise ValueError("Matched Product PPO fell below 30,000 steady SPS")
        console = (probe / "run/console.log").read_text()
        marked = "PRODUCT_HEAD_GRADIENT_MASK words=64 before_global_clip=1" in console
        if marked != (arm == "control"):
            raise ValueError("Product Q gradient mask applied to the wrong arm")
        if digest(probe / "run/initial-policy.bin") != source_policy:
            raise ValueError("Pair arm did not start from the exact Product source policy")
        phase("ppo", "end", arm=arm, steady_sps=audit["steady_sps"])
        checkpoint = probe / f"run/checkpoints/metta_generals/run/{train['steps_per_arm']:016d}.bin"
        sampler = target / "sampler.json"
        sampler.write_text(json.dumps(trial.sampler, indent=2) + "\n")
        phase("export", "start", arm=arm)
        trial.call("publish_policy_asset", ["--build", output / "build/build.json", "--training",
             probe / "run/training.json", "--checkpoint", checkpoint, "--sha256", digest(checkpoint),
             "--sampler", sampler, "--factory-source", source / "integrations/generals_fabric.py",
             "--output", target / "asset"], name="publish", seconds=600, arm="probe")
        asset = target / "asset/asset.json"
        trial.call("export_spatial_policy_bundle", ["--asset", asset,
             "--manifest-sha256", digest(asset), "--factory-source", source / "integrations/generals_fabric.py",
             "--output", target / "bundle"], name="export", seconds=600, arm="probe")
        if digest(target / "bundle/policy.bin") != digest(checkpoint):
            raise ValueError("Exported Product bundle differs from trained checkpoint")
        if arm == "control":
            verify_control(inputs / "bundles/cold", target / "bundle")
        phase("export", "end", arm=arm, checkpoint_sha256=digest(checkpoint))
        phase("serving_parity", "start", arm=arm)
        parity = probe / "trained-parity.json"
        trial.call("audit_spatial_checkpoint_serving_parity", [
             "--bundle", target / "bundle", "--replay-root", inputs / "leader-root",
             "--factory-source", source / "integrations/generals_fabric.py",
             "--output", parity], name="trained-parity", seconds=600, arm="probe")
        verify_serving_parity(parity, checkpoint, target / "bundle", source, intent)
        arm_receipts[arm] = {
            "checkpoint_sha256": digest(checkpoint),
            "bundle_manifest_sha256": digest(target / "bundle/spatial-policy.json"),
            "native_serving_parity_sha256": digest(parity),
            "training_audit_sha256": digest(probe / "training-audit.json"),
        }
        if arm == "product":
            activation = probe / "activation-gate.json"
            trial.call("audit_product_activation", [
                 "gate", "--source-bundle", inputs / "bundles/cold",
                 "--trained-bundle", target / "bundle", "--checkpoint", checkpoint,
                 "--parity-report", parity, "--views", inputs / "activation-views.npz",
                 "--output", activation], name="activation-gate", seconds=180, arm="probe")
            if json.loads(activation.read_text())["passed"] is not True:
                raise ValueError("Matched Product head did not activate")
            arm_receipts[arm]["activation_gate_sha256"] = digest(activation)
        phase("serving_parity", "end", arm=arm)
    for arm in ARMS:
        phase("evaluation", "start", arm=arm, games=dev["games_per_arm"])
        shared.call("evaluate_spatial_population", ["--bundle", output / arm / "bundle",
             "--population-build", output / "build/build.json", "--games", dev["games_per_arm"],
             "--pool-size", dev["pool_size"], "--seed", dev["map_seed"],
             "--sample-seed", dev["sample_seed"], "--output", output / ("eval-" + arm)],
             name="evaluate-" + arm, seconds=1800)
        phase("evaluation", "end", arm=arm)
    for name in ("initial_sides", "initial_state_sha256", "opponent_labels"):
        left = np.load(output / f"eval-control/{name}.npy", allow_pickle=False)
        right = np.load(output / f"eval-product/{name}.npy", allow_pickle=False)
        if len(left) != dev["games_per_arm"] or not np.array_equal(left, right):
            raise ValueError("Product development evaluations are not exactly paired: " + name)
    comparison = output / "comparison.json"
    phase("paired_analysis", "start")
    shared.call("analyze_spatial_population_pair", ["--baseline", output / "eval-control",
         "--candidate", output / "eval-product", "--seed", dev["bootstrap_seed"],
         "--bootstrap-resamples", 10000, "--output", comparison], name="compare", seconds=90)
    report = json.loads(comparison.read_text())
    broad = [cell for seats in report["by_opponent_and_seat"].values()
             for cell in seats.values() if cell["games"] >= 100]
    selected = (report["initial_state_cluster_ci95"][0] > 0
                and all(cell["paired_signed_score_delta"] >= -0.10 for cell in broad))
    (output / "COMPLETED.json").write_text(json.dumps({
        "experiment": plan["schema"], "development_only": True,
        "product_selected_for_independent_confirmation": selected,
        "comparison_sha256": digest(comparison), "plan_sha256": digest(inputs / "plan.json"),
        "qualified_probe_sha256": lineage["probe_marker_sha256"], "gpu": gpu,
        "arms": arm_receipts,
    }, indent=2) + "\n")
    phase("paired_analysis", "end", selected=selected, comparison_sha256=digest(comparison))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare")
    for name in ("probe-input", "repository", "marker", "terminal", "activation-report", "output"):
        stage.add_argument("--" + name, type=Path, required=True)
    stage.add_argument("--marker-sha256", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.probe_input, args.repository, args.marker, args.terminal,
                args.activation_report, args.marker_sha256, args.output)
    else:
        run_pair(args.input, args.output)


if __name__ == "__main__":
    main()

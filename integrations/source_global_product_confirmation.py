"""Seal and run the independent three-arm Product Classic confirmation.

`prepare` is deliberately fail-closed: it writes no context until the exact
matched development job, checkpoint exports and serving proofs all verify.
Neither command submits compute or registers a hosted policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

from integrations.analyze_spatial_population_pair import compare
from integrations.classic_contract import ENGINE_SHA256
from integrations.source_global_product_pair import verify_control
from integrations.source_global_product_probe import file_hashes
from integrations.spatial_policy_bundle import SpatialPlayerPolicy

PAIR_JOB = "job-9sump"
TERMINAL_SHA256 = None  # Pin only after successful selected development terminal is independently verified.
PLAN_SHA256 = "eb940b4b66c85e4d19fb683c48cf6bdbf0fbbee1b603d97fa0ac8feb91457362"
ARMS = ("source", "control", "product")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def audit_pair(context: Path, results: Path) -> tuple[dict, dict, Path]:
    if TERMINAL_SHA256 is None:
        raise ValueError("Exact selected job-9sump terminal is not pinned")
    inputs = context / "input"
    if file_hashes(inputs) != read(inputs / "seal.json"):
        raise ValueError("Matched pair input seal differs")
    plan_path = inputs / "plan.json"
    if digest(plan_path) != PLAN_SHA256:
        raise ValueError("Independent confirmation seed plan differs")
    plan = read(plan_path)
    reserved = plan["reserved_independent_confirmation"]
    if any(reserved.get(key) != value for key, value in {
            "map_seed": 10446701, "sample_seed": 10446703, "bootstrap_seed": 10446711}.items()):
        raise ValueError("Reserved independent seeds differ")
    terminal = read(results / "terminal.json")
    archive = results / "output.tar"
    if (digest(results / "terminal.json") != TERMINAL_SHA256
            or terminal.get("job_id") != PAIR_JOB or terminal.get("status") != "succeeded"
            or terminal.get("restarts_used") != 0
            or terminal.get("checks", {}).get("success_verdict") != "passed"
            or terminal.get("artifact", {}).get("state") != "ready"
            or terminal["artifact"]["size_bytes"] != archive.stat().st_size
            or terminal["artifact"]["sha256"] != digest(archive)):
        raise ValueError("Exact matched development job is not a verified success")
    # Bind extracted checkpoints, reports and episode arrays to the provider archive.
    with tarfile.open(archive) as contents:
        for member in contents:
            if member.isfile():
                with contents.extractfile(member) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest(results / "extracted" / member.name) != expected:
                    raise ValueError("Extracted development artifact differs: " + member.name)
    output = results / "extracted/generals"
    completed = read(output / "COMPLETED.json")
    comparison = output / "comparison.json"
    if (completed.get("experiment") != plan["schema"]
            or completed.get("development_only") is not True
            or completed.get("product_selected_for_independent_confirmation") is not True
            or completed.get("plan_sha256") != PLAN_SHA256
            or completed.get("comparison_sha256") != digest(comparison)
            or completed.get("qualified_probe_sha256") != digest(inputs / "qualified-probe-marker.json")):
        raise ValueError("Product failed or changed its preregistered development gate")
    dev = plan["development"]
    recomputed = compare(output / "eval-control", output / "eval-product",
                         seed=dev["bootstrap_seed"], resamples=10000)
    if recomputed != read(comparison) or recomputed["games"] != dev["games_per_arm"]:
        raise ValueError("Development comparison differs from retained episode arrays")
    broad = [cell for seats in recomputed["by_opponent_and_seat"].values()
             for cell in seats.values() if cell["games"] >= 100]
    if (recomputed["initial_state_cluster_ci95"][0] <= 0
            or not broad or any(cell["paired_signed_score_delta"] < -0.10 for cell in broad)):
        raise ValueError("Recomputed Product development gate failed")
    return plan, terminal, output


def effective_sampler(sampler: dict) -> dict:
    return {"full_action_temperature": 1.0, "route_half_weight": 0.0, **sampler}


def audit_bundle(bundle: Path, checkpoint: Path, proof: Path) -> dict:
    policy = SpatialPlayerPolicy(bundle)
    checkpoint_sha = digest(checkpoint)
    bundle_sha = digest(bundle / "spatial-policy.json")
    report = read(proof)
    parity_errors = (report.get("max_logit_difference"),
                     report.get("max_action_probability_difference"),
                     report.get("max_rollout_transform_difference"))
    if (digest(bundle / "policy.bin") != checkpoint_sha
            or policy.asset.metadata["policy_sha256"] != checkpoint_sha
            or report.get("checkpoint_sha256") != checkpoint_sha
            or report.get("bundle_manifest_sha256") != bundle_sha
            or report.get("factory_source_sha256") != policy.asset.metadata["factory_source_sha256"]
            or report.get("engine_sha256") != ENGINE_SHA256
            or report.get("public_states", 0) < 16
            or report.get("hosted_games", 0) < 4
            or report.get("matching_top_actions") != report["public_states"]
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                   for value in parity_errors)
            or parity_errors[0] > 2e-5 or parity_errors[1] > 1e-5
            or parity_errors[2] > 1e-5):
        raise ValueError("Checkpoint, Product bundle or native-serving parity differs")
    return {"checkpoint_sha256": checkpoint_sha, "bundle_manifest_sha256": bundle_sha,
            "asset_manifest_sha256": digest(bundle / "asset.json"),
            "weights_sha256": digest(bundle / "weights.npz"), "parity_sha256": digest(proof),
            "sampler": policy.asset.metadata["sampler"],
            "training_seeds": policy.asset.metadata["training_seeds"]}


def prepare(context: Path, results: Path, source_proof: Path, repository: Path, target: Path) -> dict:
    if target.exists():
        raise FileExistsError(target)
    context, results, repository = (path.resolve(strict=True) for path in (context, results, repository))
    plan, terminal, output = audit_pair(context, results)
    comparison = output / "comparison.json"
    completed = read(output / "COMPLETED.json")
    proofs = {"source": source_proof, **{arm: output / arm / "probe/trained-parity.json"
                                       for arm in ("control", "product")}}
    if digest(source_proof) != plan["required_probe_native_source_parity_sha256"]:
        raise ValueError("Source parity differs from qualified probe")
    bundles = {"source": context / "input/bundles/cold",
               **{arm: output / arm / "bundle" for arm in ("control", "product")}}
    checkpoints = {"source": context / "input/assets/cold/policy.bin",
                   **{arm: output / arm / "probe/run/checkpoints/metta_generals/run/0000000016777216.bin"
                      for arm in ("control", "product")}}
    identity = {arm: audit_bundle(bundles[arm], checkpoints[arm], proofs[arm]) for arm in ARMS}
    if (identity["source"]["checkpoint_sha256"] != plan["source_product_policy_sha256"]
            or not all(effective_sampler(identity[arm]["sampler"]) == effective_sampler(identity["source"]["sampler"]) for arm in ARMS)
            or any(plan["reserved_independent_confirmation"]["map_seed"] in identity[arm]["training_seeds"]
                   for arm in ARMS)):
        raise ValueError("Source policy, sampler or independent training lineage differs")
    for arm in ("control", "product"):
        receipt = completed["arms"][arm]
        if (receipt["checkpoint_sha256"] != identity[arm]["checkpoint_sha256"]
                or receipt["bundle_manifest_sha256"] != identity[arm]["bundle_manifest_sha256"]
                or receipt["native_serving_parity_sha256"] != identity[arm]["parity_sha256"]
                or receipt["training_audit_sha256"] != digest(output / arm / "probe/training-audit.json")):
            raise ValueError("Development arm receipt differs: " + arm)
    activation = output / "product/probe/activation-gate.json"
    if digest(activation) != completed["arms"]["product"]["activation_gate_sha256"]:
        raise ValueError("Product activation receipt differs")
    from integrations.audit_product_activation import audit
    recovered = audit(bundles["source"], bundles["product"], checkpoints["product"],
                      proofs["product"], context / "input/activation-views.npz")
    if recovered != read(activation) or recovered["passed"] is not True:
        raise ValueError("Trained Product activation does not reproduce")
    verify_control(bundles["source"], bundles["control"])
    source_q = SpatialPlayerPolicy(bundles["source"]).weights["product_action_kernel"]
    if not np.array_equal(source_q, np.zeros((8, 8), np.float32)):
        raise ValueError("Transplanted source Product head is not zero")
    build = output / "build/build.json"
    frozen = read(build)["config"]["python_environment"]["options"]
    if (frozen["opponent_weights"] != plan["opponent_weights"]
            or len(frozen["frozen_bundles"]) != 10
            or frozen["scripted_opponents"] != ["expander_harvester", "sentinel", "classic_siege_padded"]):
        raise ValueError("Confirmation population differs from matched development")
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip():
        raise ValueError("Confirmation source must be committed before packaging")
    if digest(repository / "integrations/source_global_product_pair_plan.json") != PLAN_SHA256:
        raise ValueError("Confirmation source plan differs")
    if digest(repository / "integrations/generals_fabric.py") != SpatialPlayerPolicy(
            bundles["product"]).asset.metadata["factory_source_sha256"]:
        raise ValueError("Confirmation source Product factory differs from trained asset")
    docker = (context / "Dockerfile").read_text()
    old = '"integrations.source_global_product_pair", "run"'
    if docker.count(old) != 1:
        raise ValueError("Matched Dockerfile entrypoint differs")
    target.mkdir(parents=True)
    shutil.copytree(context / "framework-source", target / "framework-source")
    shutil.copy2(context / "requirements.lock", target / "requirements.lock")
    (target / "Dockerfile").write_text(docker.replace(old, '"integrations.source_global_product_confirmation", "run"'))
    inputs = target / "input"
    inputs.mkdir()
    (inputs / "source").mkdir()
    archive = subprocess.Popen(["git", "-C", str(repository), "archive", revision], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-x", "-C", str(inputs / "source")], stdin=archive.stdout, check=True)
    finally:
        assert archive.stdout is not None
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError("Confirmation source archive failed")
    shutil.copytree(context / "input/bundles/frozen", inputs / "bundles/frozen")
    for arm in ARMS:
        shutil.copytree(bundles[arm], inputs / "confirmation/bundles" / arm)
        (inputs / "confirmation/proofs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(proofs[arm], inputs / "confirmation/proofs" / (arm + ".json"))
    shutil.copy2(activation, inputs / "confirmation/product-activation.json")
    shutil.copy2(build, inputs / "confirmation/build.json")
    shutil.copy2(comparison, inputs / "confirmation/development-comparison.json")
    shutil.copy2(output / "COMPLETED.json", inputs / "confirmation/development-completed.json")
    shutil.copy2(context / "input/plan.json", inputs / "confirmation/plan.json")
    intent = {"schema": "generals-product-logical-independent-confirmation-v1",
              "source_revision": revision, "pair_job_id": PAIR_JOB,
              "pair_archive_sha256": terminal["artifact"]["sha256"],
              "pair_terminal_sha256": TERMINAL_SHA256,
              "product_activation_sha256": digest(activation),
              "development_completed_sha256": digest(output / "COMPLETED.json"),
              "development_comparison_sha256": digest(comparison),
              "plan_sha256": PLAN_SHA256,
              "population_build_sha256": digest(build),
              "games_per_arm": 4096, "pool_size": 4096,
              "map_seed": 10446701, "sample_seed": 10446703,
              "bootstrap_seed": 10446711, "arms": identity,
              "gate": "Both Product-control and Product-source clustered 95% lower bounds >0; no opponent-seat stratum with >=100 games below -0.10 versus either comparator"}
    (inputs / "confirmation/intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    for path in (target, *target.rglob("*")):
        if path.is_symlink():
            raise ValueError("Confirmation context contains a symlink")
        path.chmod(stat.S_IMODE(path.stat().st_mode) | (0o555 if path.is_dir() else 0o444))
    (inputs / "seal.json").write_text(json.dumps(file_hashes(inputs), indent=2) + "\n")
    (target / "seal.json").write_text(json.dumps(file_hashes(target), indent=2) + "\n")
    return intent


def run(inputs: Path, output: Path) -> None:
    inputs = inputs.resolve(strict=True)
    if file_hashes(inputs) != read(inputs / "seal.json"):
        raise ValueError("Confirmation input seal differs")
    intent = read(inputs / "confirmation/intent.json")
    plan = read(inputs / "confirmation/plan.json")
    if (intent["plan_sha256"] != PLAN_SHA256
            or any(intent[key] != plan["reserved_independent_confirmation"][key]
                   for key in ("map_seed", "sample_seed", "bootstrap_seed"))
            or intent["pair_job_id"] != PAIR_JOB or intent["pair_terminal_sha256"] != TERMINAL_SHA256
            or digest(inputs / "confirmation/product-activation.json") != intent["product_activation_sha256"]
            or intent["games_per_arm"] != 4096 or intent["pool_size"] != 4096):
        raise ValueError("Independent confirmation intent differs")
    for arm in ARMS:
        bundle = inputs / "confirmation/bundles" / arm
        proof = inputs / "confirmation/proofs" / (arm + ".json")
        checked = audit_bundle(bundle, bundle / "policy.bin", proof)
        if checked != intent["arms"][arm]:
            raise ValueError("Confirmation arm identity differs: " + arm)
    if digest(inputs / "confirmation/build.json") != intent["population_build_sha256"]:
        raise ValueError("Confirmation population build differs")
    from integrations.cuda_runtime_binding import configure
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
    configure()
    gpu = verify_gpu_idle(visible_gpu_identity())
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    import jax
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Independent Classic evaluation requires GPU")
    output.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        command = [sys.executable, "-m", "integrations.evaluate_spatial_population",
                   "--bundle", str(inputs / "confirmation/bundles" / arm),
                   "--population-build", str(inputs / "confirmation/build.json"),
                   "--games", "4096", "--pool-size", "4096",
                   "--seed", str(intent["map_seed"]), "--sample-seed", str(intent["sample_seed"]),
                   "--output", str(output / ("eval-" + arm))]
        with (output / ("eval-" + arm + ".log")).open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        record = read(output / ("eval-" + arm) / "evaluation.json")
        strata = record["by_opponent_and_seat"]
        if (record["checkpoint_sha256"] != intent["arms"][arm]["checkpoint_sha256"]
                or record["games"] != 4096 or record["pool_size"] != 4096
                or record["seed"] != intent["map_seed"]
                or record["sample_seed"] != intent["sample_seed"]
                or record["coworld_classic_rules"] is not True
                or record["opponent_weights"] != plan["opponent_weights"]
                or len(strata) != 13
                or any(set(seats) != {"0", "1"} or seats["0"]["games"] != seats["1"]["games"]
                       or seats["0"]["games"] <= 0 for seats in strata.values())):
            raise ValueError("Confirmation arm checkpoint, rules or seat coverage differs: " + arm)
    arrays = ("initial_state_sha256", "initial_sides", "opponent_labels")
    for name in arrays:
        source = np.load(output / "eval-source" / (name + ".npy"), allow_pickle=False)
        if len(source) != 4096 or any(not np.array_equal(source, np.load(
                output / ("eval-" + arm) / (name + ".npy"), allow_pickle=False))
                for arm in ("control", "product")):
            raise ValueError("Three confirmation arms differ in initial " + name)
    reports = {}
    for arm in ("control", "source"):
        result = compare(output / ("eval-" + arm), output / "eval-product",
                         seed=intent["bootstrap_seed"], resamples=10000)
        reports["product_minus_" + arm] = result
        (output / ("product-minus-" + arm + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    def passes(report):
        cells = [cell for seats in report["by_opponent_and_seat"].values()
                 for cell in seats.values() if cell["games"] >= 100]
        return (report["games"] == 4096 and report["initial_state_cluster_ci95"][0] > 0
                and bool(cells) and all(cell["paired_signed_score_delta"] >= -0.10 for cell in cells))
    (output / "COMPLETED.json").write_text(json.dumps({
        "schema": intent["schema"], "development_only": False, "hosted": False,
        "independent_confirmation_passes": all(map(passes, reports.values())),
        "intent_sha256": digest(inputs / "confirmation/intent.json"),
        "comparisons": {name: digest(output / (name.replace("_", "-") + ".json"))
                        for name in reports},
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("prepare")
    for name in ("pair-context", "pair-results", "parity-source", "repository", "output"):
        stage.add_argument("--" + name, type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.pair_context, args.pair_results,
                         args.parity_source,
                         args.repository, args.output)
        print(json.dumps(result, sort_keys=True))
    else:
        run(args.input, args.output)


if __name__ == "__main__":
    main()

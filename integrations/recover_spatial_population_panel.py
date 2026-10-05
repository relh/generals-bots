"""Recover one interrupted matched panel without retraining or overwriting evidence.

Run under the usual bounded GPU step. The executor replaces this process so
the step retains ownership of the evaluator and its termination lifecycle.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from integrations.analyze_spatial_population_pair import compare
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_plan(reference, interrupted, bundle, population_build, output,
                  checkpoint_sha256, bundle_manifest_sha256):
    paths = [Path(p).resolve() for p in (reference, interrupted, bundle, population_build, output)]
    reference, interrupted, bundle, population_build, output = paths
    if output.exists():
        raise ValueError("Recovery output already exists; preserve it and choose a new path")
    if any(output == p or p in output.parents or output in p.parents for p in paths[:-1]):
        raise ValueError("Recovery output must be separate from retained inputs")
    if (interrupted / "evaluation.json").exists() or (interrupted / "outcomes.npy").exists():
        raise ValueError("Interrupted panel contains outcomes; reconcile rather than replay it")
    # Validate counts, per-opponent seats and every completed outcome before reuse.
    compare(reference, reference, resamples=1)
    record = json.loads((reference / "evaluation.json").read_text())
    if sha256(population_build) != record["population_build_sha256"]:
        raise ValueError("Population build differs from the completed reference")
    if sha256(bundle / "policy.bin") != checkpoint_sha256:
        raise ValueError("Recovery checkpoint differs from its pinned identity")
    if sha256(bundle / "spatial-policy.json") != bundle_manifest_sha256:
        raise ValueError("Recovery bundle manifest differs from its pinned identity")
    policy = SpatialPlayerPolicy(bundle)
    if policy.action_mode != "structured_sample":
        raise ValueError("Recovery requires the exact structured serving sampler")
    initial_hashes = {}
    for name in ("initial_state_sha256", "initial_sides", "opponent_labels"):
        old = interrupted / (name + ".npy")
        expected = reference / old.name
        if not np.array_equal(np.load(old, allow_pickle=False), np.load(expected, allow_pickle=False)):
            raise ValueError(f"Interrupted panel differs from reference in {name}")
        initial_hashes[name] = sha256(old)
    arguments = [sys.executable, "-m", "integrations.evaluate_spatial_population",
                 "--bundle", str(bundle), "--population-build", str(population_build),
                 "--output", str(output)]
    for key in ("games", "pool_size", "seed", "sample_seed"):
        arguments += ["--" + key.replace("_", "-"), str(record[key])]
    if "destination_audit" in record:
        arguments.append("--destination-audit")
    return dict(scope="One missing panel only; retained inputs remain read-only",
                reference=str(reference), interrupted=str(interrupted), output=str(output),
                checkpoint_sha256=checkpoint_sha256,
                bundle_manifest_sha256=bundle_manifest_sha256,
                population_build_sha256=record["population_build_sha256"],
                initial_array_sha256=initial_hashes, arguments=arguments,
                required_followup="Compare recovered panel against the completed panels after verified collection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "interrupted", "bundle", "population-build", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    receipt = args.receipt.resolve()
    for source in (args.reference, args.interrupted, args.bundle, args.population_build):
        source = source.resolve()
        if receipt == source or source in receipt.parents:
            raise ValueError("Recovery receipt must be separate from retained inputs")
    plan = recovery_plan(args.reference, args.interrupted, args.bundle, args.population_build,
                         args.output, args.checkpoint_sha256, args.bundle_manifest_sha256)
    with args.receipt.open("x") as target:
        json.dump(plan, target, indent=2)
        target.write("\n")
    if not args.plan_only:
        os.execv(sys.executable, plan["arguments"])


if __name__ == "__main__":
    main()

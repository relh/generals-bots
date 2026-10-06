"""One-time, hash-checked source checkpoint transplant into the rank-8 policy.

The source factory and source direct evaluator are supplied as archived files;
this command does not carry an old model or serving fallback in the fork.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import jax
import numpy as np
from metta_training.model_config import FabricConfig
from metta_training.native_build import fabric_fingerprint
from metta_training.native_fabric import NativeFabricPolicy

from integrations.direct_spatial_optimization import DirectSpatial
from integrations.native_spatial_asset import abi_digest, canonical_json, sha256, write_asset


def archived_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import archived source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parameter_blocks(policy):
    leaves = jax.tree_util.tree_flatten_with_path(policy.buffers.fn.params(policy.buffers.template))[0]
    blocks = {}
    for (path, _), block in zip(leaves, policy.buffers.parameters, strict=True):
        name = jax.tree_util.keystr(path)
        if name in blocks:
            raise ValueError(f"Repeated parameter path: {name}")
        blocks[name] = block
    return blocks


def transplant(old_manifest: Path, old_manifest_sha256: str, old_factory: Path, old_direct: Path,
               output: Path, seed: int):
    if output.exists():
        raise FileExistsError(output)
    raw = old_manifest.read_bytes()
    if sha256(raw) != old_manifest_sha256:
        raise ValueError("Archived asset manifest hash differs")
    metadata = json.loads(raw)
    if metadata["factory_source_sha256"] != sha256(old_factory.read_bytes()):
        raise ValueError("Archived factory source hash differs")
    old_policy_bytes = old_manifest.with_name("policy.bin").read_bytes()
    if metadata["policy_sha256"] != sha256(old_policy_bytes):
        raise ValueError("Archived policy hash differs")
    old_values = np.frombuffer(old_policy_bytes, "<f4")
    if old_values.size != metadata["parameter_count"] or not np.isfinite(old_values).all():
        raise ValueError("Archived policy length or finite check failed")
    if metadata["fabric"]["factory"] != "integrations.generals_fabric:two_stage_tied_local_action_policy":
        raise ValueError("Archived factory is not the selected spatial source")
    source_module = archived_module("archived_generals_fabric", old_factory)
    old_direct_module = archived_module("archived_direct_spatial", old_direct)
    if not hasattr(source_module, "two_stage_tied_local_action_policy"):
        raise ValueError("Archived factory is missing")
    source_config = dict(metadata["fabric"], factory="archived_generals_fabric:two_stage_tied_local_action_policy")
    target_config = metadata["fabric"]
    old_native = NativeFabricPolicy(json.dumps(source_config))
    new_native = NativeFabricPolicy(json.dumps(target_config))
    if (old_native.buffers.parameter_words != old_values.size
            or new_native.buffers.parameter_words != old_values.size + 576):
        raise ValueError("Source or target checkpoint geometry differs")
    old_blocks = parameter_blocks(old_native)
    new_blocks = parameter_blocks(new_native)
    if not old_blocks.keys() <= new_blocks.keys():
        raise ValueError(f"Old parameter paths vanished: {sorted(old_blocks.keys() - new_blocks.keys())}")
    if len(new_blocks) - len(old_blocks) != 3:
        raise ValueError("Expected precisely three source-conditioned parameter blocks")
    new_values = np.frombuffer(new_native.initialize(seed), "<f4").copy()
    for path, old_block in old_blocks.items():
        new_block = new_blocks[path]
        if old_block.shape != new_block.shape or old_block.size != new_block.size:
            raise ValueError(f"Existing parameter shape drifted: {path}")
        new_values[new_block.offset : new_block.offset + new_block.size] = (
            old_values[old_block.offset : old_block.offset + old_block.size]
        )
    old_model = old_direct_module.DirectSpatial(old_native)
    new_model = DirectSpatial(new_native)
    if not np.all(new_values[new_model.product_action_kernel] == 0):
        raise ValueError("Residual head must initialize to exact zero")
    random = np.random.default_rng(17290391)
    observations = random.normal(0, 0.2, (2, 7056)).astype(np.float32)
    old_logits = np.asarray(old_model.forward(old_values, observations))
    new_logits = np.asarray(new_model.forward(new_values, observations))
    if not np.array_equal(old_logits, new_logits):
        raise ValueError(f"Source logits are not bitwise equal at initialization: "
                         f"max delta={np.max(np.abs(old_logits - new_logits))}")
    output.mkdir(parents=True)
    policy = output / "transplanted-policy.bin"
    policy.write_bytes(new_values.astype("<f4", copy=False).tobytes())
    factory_source = Path(__file__).with_name("generals_fabric.py")
    proof = {
        "schema": "generals-source-global-product-transplant-v1",
        "source_manifest_sha256": old_manifest_sha256,
        "source_factory_sha256": metadata["factory_source_sha256"],
        "source_policy_sha256": metadata["policy_sha256"],
        "source_direct_sha256": sha256(old_direct.read_bytes()),
        "target_factory_sha256": sha256(factory_source.read_bytes()),
        "target_policy_sha256": sha256(policy.read_bytes()),
        "source_parameter_words": old_values.size,
        "target_parameter_words": new_values.size,
        "copied_parameter_blocks": len(old_blocks),
        "added_parameter_paths": sorted(new_blocks.keys() - old_blocks.keys()),
        "zero_head": True,
        "bitwise_equal_source_logits": True,
        "parity_observation_seed": 17290391,
        "parity_observations": len(observations),
        "initialization_seed": seed,
    }
    receipt = output / "transplant-proof.json"
    receipt.write_bytes(canonical_json(proof) + b"\n")
    provenance = {
        "operation": "architecture_init", "reinforcement_learning_steps_added": 0,
        "ancestors": {
            "source_asset_manifest": old_manifest_sha256,
            "source_policy": metadata["policy_sha256"],
            "transplant_proof": sha256(receipt.read_bytes()),
        },
        "opaque": {"source_agent_steps": metadata["provenance"].get("reinforcement_learning_steps_added", 0)},
    }
    asset_path = write_asset(
        output / "asset", fabric=target_config,
        factory_source_sha256=proof["target_factory_sha256"],
        model_sha256=fabric_fingerprint(FabricConfig.model_validate(target_config)),
        abi_sha256=abi_digest(new_native), policy=policy, sampler=metadata["sampler"],
        provenance=provenance, training_seeds=sorted(set(metadata["training_seeds"] + [seed])),
    )
    return {"asset": str(asset_path), "asset_sha256": sha256(asset_path.read_bytes()),
            "proof": str(receipt), "proof_sha256": sha256(receipt.read_bytes()), **proof}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--old-manifest-sha256", required=True)
    parser.add_argument("--old-factory", type=Path, required=True)
    parser.add_argument("--old-direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10942311)
    args = parser.parse_args()
    print(json.dumps(transplant(args.old_manifest, args.old_manifest_sha256, args.old_factory,
                                args.old_direct, args.output, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()

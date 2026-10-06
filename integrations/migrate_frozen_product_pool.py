"""One-time zero-head migration of the selected Classic frozen opponent pool."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from metta_training.model_config import FabricConfig
from metta_training.native_build import fabric_fingerprint
from metta_training.native_fabric import NativeFabricPolicy

from integrations.direct_spatial_optimization import DirectSpatial
from integrations.native_spatial_asset import abi_digest, canonical_json, sha256, write_asset
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import encode_wire_observation
from integrations.spatial_policy_bundle import SpatialPlayerPolicy, structured_action_probabilities
from integrations.transplant_source_global_product import archived_module, parameter_blocks

SOURCE_FACTORY_SHA256 = "48767fb4ee333ae0b1a02ae644fbdf6f52f7f6df6c90c97ab3fc3888ba0c0d8a"
SOURCE_DIRECT_SHA256 = "f9aaf04769b01f024b21a016fc2211e7403ae0824522a9c849f189bfb18931d3"
SOURCE_POLICY_SHA256 = "f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14"
TARGET_POLICY_SHA256 = "9c55dc3b167746f0153f5de50afaa6cba9cfb4a8858107d524c05d988c9147e9"


def public_views():
    values, masks = [], []
    for seed in (482112, 482123):
        match = Match(seed)
        for _ in range(64):
            match.advance([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
        for seat in (0, 1):
            observation, legal = encode_wire_observation(match.observation(seat))
            if legal.sum() < 2:
                raise ValueError("Seeded Classic state has too few legal moves")
            values.append(observation)
            masks.append(legal)
    return np.stack(values), np.stack(masks)


def export_weights(model, parameters):
    weights = {}
    for name in ("input_kernel", "context_kernel", "action_kernel", "product_local_kernel",
                 "product_global_kernel", "product_action_kernel"):
        indices = getattr(model, name)
        weights[name] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    for name in ("local_weight", "local_bias", "context_weight", "context_bias"):
        weights[name] = parameters[getattr(model, name)[0]]
    for name in ("global_weight", "global_bias", "global_kernel", "readout_kernel", "output_weight", "output_bias"):
        weights[name] = parameters[getattr(model, name)]
    for i, (source, indices) in enumerate(model.priors):
        weights[f"prior_source_{i}"] = source
        weights[f"prior_weight_{i}"] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    return weights


def seal_bundle(path):
    files = {name: sha256((path / name).read_bytes()) for name in ("asset.json", "policy.bin", "weights.npz")}
    manifest = path / "spatial-policy.json"
    manifest.write_bytes(canonical_json({"schema": "generals-spatial-policy-v1", "files": files}) + b"\n")
    return sha256(manifest.read_bytes())


def run(source_pool: Path, source_asset: Path, target_bundle: Path,
        old_factory: Path, old_direct: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    if (sha256(old_factory.read_bytes()) != SOURCE_FACTORY_SHA256
            or sha256(old_direct.read_bytes()) != SOURCE_DIRECT_SHA256):
        raise ValueError("Archived source code differs")
    archive = archived_module("archived_generals_fabric", old_factory)
    old_direct_module = archived_module("archived_direct_spatial", old_direct)
    if not hasattr(archive, "two_stage_tied_local_action_policy"):
        raise ValueError("Archived spatial factory is missing")
    source_metadata = json.loads(source_asset.read_text())
    target_metadata = json.loads((target_bundle / "asset.json").read_text())
    if (source_metadata["policy_sha256"] != SOURCE_POLICY_SHA256
            or target_metadata["policy_sha256"] != TARGET_POLICY_SHA256):
        raise ValueError("Source or Product selected policy differs")
    if target_metadata["provenance"]["operation"] != "architecture_init":
        raise ValueError("Product source must be zero-head architecture initialization")
    views, legal_masks = public_views()
    output.mkdir(parents=True)
    bundle_root = output / "bundles/frozen"
    bundle_root.mkdir(parents=True)
    receipts = []
    source_sha = sha256(Path(__file__).with_name("generals_fabric.py").read_bytes())
    for radius, indices in ((2.01, [0]), (1.01, list(range(1, 10)))):
        example = json.loads((source_pool / str(indices[0]) / "asset.json").read_text())
        config = example["fabric"]
        if config["options"]["context_radius"] != radius:
            raise ValueError("Opponent radius differs")
        archived_config = dict(config, factory="archived_generals_fabric:two_stage_tied_local_action_policy")
        old_native = NativeFabricPolicy(json.dumps(archived_config))
        new_native = NativeFabricPolicy(json.dumps(config))
        old_model = old_direct_module.DirectSpatial(old_native)
        new_model = DirectSpatial(new_native)
        old_blocks, new_blocks = parameter_blocks(old_native), parameter_blocks(new_native)
        if not old_blocks.keys() <= new_blocks.keys() or len(new_blocks) - len(old_blocks) != 3:
            raise ValueError("Old parameter topology did not embed in Product topology")
        abi_sha = abi_digest(new_native)
        model_sha = fabric_fingerprint(FabricConfig.model_validate(config))
        initialized = np.frombuffer(new_native.initialize(10942311), "<f4").copy()
        for index in indices:
            old_dir = source_pool / str(index)
            old_manifest = old_dir / "asset.json"
            old_meta = json.loads(old_manifest.read_text())
            if (old_meta["factory_source_sha256"] != SOURCE_FACTORY_SHA256
                    or old_meta["fabric"] != config
                    or sha256((old_dir / "policy.bin").read_bytes()) != old_meta["policy_sha256"]):
                raise ValueError(f"Frozen source identity differs at slot {index}")
            old_values = np.frombuffer((old_dir / "policy.bin").read_bytes(), "<f4")
            new_values = initialized.copy()
            for path, old_block in old_blocks.items():
                new_block = new_blocks[path]
                if old_block.shape != new_block.shape or old_block.size != new_block.size:
                    raise ValueError(f"Frozen source parameter drift: {path}")
                new_values[new_block.offset:new_block.offset + new_block.size] = (
                    old_values[old_block.offset:old_block.offset + old_block.size]
                )
            if not np.all(new_values[new_model.product_action_kernel] == 0):
                raise ValueError("Frozen residual head is nonzero")
            old_logits = np.asarray(old_model.forward(old_values, views))
            new_logits = np.asarray(new_model.forward(new_values, views))
            if not np.array_equal(old_logits, new_logits):
                raise ValueError(f"Frozen opponent {index} changed direct logits")
            target = bundle_root / str(index)
            if index == 0:
                if sha256(new_values.astype("<f4").tobytes()) != TARGET_POLICY_SHA256:
                    raise ValueError("Selected Product source policy changed")
                shutil.copytree(target_bundle, target)
            else:
                raw = output / f"slot-{index}.bin"
                raw.write_bytes(new_values.astype("<f4").tobytes())
                provenance = {
                    "operation": "architecture_init", "reinforcement_learning_steps_added": 0,
                    "ancestors": {"source_asset_manifest": sha256(old_manifest.read_bytes()),
                                  "source_policy": old_meta["policy_sha256"]},
                }
                write_asset(target, fabric=config, factory_source_sha256=source_sha,
                            model_sha256=model_sha, abi_sha256=abi_sha, policy=raw,
                            sampler=old_meta["sampler"], provenance=provenance,
                            training_seeds=sorted(set(old_meta["training_seeds"] + [10942311])))
                np.savez_compressed(target / "weights.npz", **export_weights(new_model, new_values))
                seal_bundle(target)
                raw.unlink()
            portable = SpatialPlayerPolicy(target)
            portable_logits = portable.forward(views)
            max_logit_delta = float(np.max(np.abs(old_logits - portable_logits)))
            max_acting_delta = 0.0
            for state, legal, before, after in zip(views, legal_masks, old_logits, portable_logits, strict=True):
                s = old_meta["sampler"]
                args = dict(observations=state, neutral_route_bias=s.get("neutral_route_bias", 0),
                            weak_owned_route_penalty=s.get("weak_owned_route_penalty", 0),
                            doomed_attack_route_penalty=s.get("doomed_attack_route_penalty", 0))
                old_probs = structured_action_probabilities(before, legal, s["move_temperature"],
                                                            s["split_temperature"], **args)
                new_probs = structured_action_probabilities(after, legal, s["move_temperature"],
                                                            s["split_temperature"], **args)
                max_acting_delta = max(max_acting_delta, float(np.max(np.abs(old_probs - new_probs))))
            if max_logit_delta > 2e-5 or max_acting_delta > 2e-5:
                raise ValueError(f"Frozen opponent {index} serving or acting drift")
            receipts.append({
                "slot": index, "radius": radius, "old_asset_sha256": sha256(old_manifest.read_bytes()),
                "old_bundle_sha256": sha256((old_dir / "spatial-policy.json").read_bytes()),
                "old_policy_sha256": old_meta["policy_sha256"],
                "new_asset_sha256": sha256((target / "asset.json").read_bytes()),
                "new_bundle_sha256": sha256((target / "spatial-policy.json").read_bytes()),
                "new_policy_sha256": sha256((target / "policy.bin").read_bytes()),
                "sampler_unchanged": old_meta["sampler"] == portable.asset.metadata["sampler"],
                "direct_logits_bitwise_equal": True,
                "portable_logits_max_abs": max_logit_delta,
                "acting_probabilities_max_abs": max_acting_delta,
            })
    if len(receipts) != 10 or any(not row["sampler_unchanged"] for row in receipts):
        raise ValueError("Original ten frozen opponents were not preserved")
    report = {"schema": "generals-product-frozen-pool-migration-v1",
              "source_factory_sha256": SOURCE_FACTORY_SHA256,
              "target_factory_sha256": source_sha,
              "public_map_seeds": [482112, 482123], "public_turn": 64,
              "public_observations": len(views), "frozen_opponents": receipts}
    receipt = output / "migration-receipt.json"
    receipt.write_bytes(canonical_json(report) + b"\n")
    return {"receipt": str(receipt), "receipt_sha256": sha256(receipt.read_bytes()), "opponents": len(receipts)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pool", type=Path, required=True)
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--target-bundle", type=Path, required=True)
    parser.add_argument("--old-factory", type=Path, required=True)
    parser.add_argument("--old-direct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source_pool, args.source_asset, args.target_bundle,
                         args.old_factory, args.old_direct, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

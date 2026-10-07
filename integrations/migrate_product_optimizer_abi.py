"""Rebind the Product source and frozen pool to a changed optimizer ABI.

The policy graph and checkpoint bytes must remain identical. This migration is
for the experiment in which Product U, V, and Q receive logical Muon matrices.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from metta_training.model_config import FabricConfig
from metta_training.native_build import fabric_fingerprint
from metta_training.native_fabric import NativeFabricPolicy

from integrations.direct_spatial_optimization import DirectSpatial
from integrations.export_spatial_policy_bundle import export_bundle
from integrations.migrate_frozen_product_pool import public_views
from integrations.native_spatial_asset import (
    abi_descriptor, abi_digest, canonical_json, load_asset, write_asset,
)
from integrations.spatial_policy_bundle import SpatialPlayerPolicy, structured_action_probabilities
from integrations.transplant_source_global_product import archived_module


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_seal(root: Path) -> None:
    sealed = read(root / "seal.json")
    actual = {str(path.relative_to(root)): digest(path)
              for path in sorted(root.rglob("*")) if path.is_file() and path != root / "seal.json"}
    require(actual == sealed, "Prior Product input differs from its complete seal")


def optimizer_identity(old_input: Path, old_repository: Path, repository: Path):
    plan = read(old_input / "plan.json")
    old_cold = old_input / "assets/cold/asset.json"
    old_meta = read(old_cold)
    verify_seal(old_input)
    require(digest(old_cold) == plan["source_product_asset_sha256"] and
            digest(old_cold.with_name("policy.bin")) == plan["source_product_policy_sha256"] and
            digest(old_input / "pool-migration-receipt.json") ==
            plan["pool_migration_receipt_sha256"], "Prior Product source differs")
    factory = repository / "integrations/generals_fabric.py"
    for name in ("spatial_action_sampling.py", "spatial_exploration.py",
                 "native_spatial_asset.py", "export_spatial_policy_bundle.py"):
        relative = Path("integrations") / name
        require(digest(old_repository / relative) == digest(repository / relative),
                f"Shared serving or asset code changed with Product repair: {name}")
    require(digest(old_repository / "integrations/generals_fabric.py") ==
            old_meta["factory_source_sha256"], "Prior factory source identity differs")
    expected = {"product_local": [8, 32], "product_global": [32, 8],
                "product_action": [8, 8]}
    topologies, directs = {}, {}
    old_assets = [old_cold] + [old_input / f"bundles/frozen/{slot}/asset.json"
                               for slot in range(10)]
    for old_asset in old_assets:
        metadata = read(old_asset)
        key = metadata["model_sha256"]
        require(metadata["factory_source_sha256"] == old_meta["factory_source_sha256"],
                "Frozen factory source differs")
        if key in topologies:
            topology = topologies[key]
            require(metadata["abi_sha256"] == topology["old_abi_sha256"] and
                    metadata["fabric"] == topology["fabric"],
                    "Frozen model/ABI identity differs within topology")
            continue
        with jax.default_device(jax.devices("cpu")[0]):
            native = NativeFabricPolicy(json.dumps(metadata["fabric"]))
            directs[key] = DirectSpatial(native)
            descriptor = abi_descriptor(native)
        blocks = {block["name"]: block for block in descriptor["optimizer_blocks"]}
        for name, shape in expected.items():
            require(blocks[name]["shape"] == shape,
                    f"Product Muon block is not a matrix: {name}")
        require(native.buffers.parameter_words == metadata["parameter_count"],
                "Product parameter storage changed")
        new_model = fabric_fingerprint(FabricConfig.model_validate(metadata["fabric"]))
        new_abi = abi_digest(native)
        require(new_model != key and new_abi != metadata["abi_sha256"],
                "Product factory/model/ABI did not change")
        topologies[key] = {
            "fabric": metadata["fabric"],
            "old_model_sha256": key, "new_model_sha256": new_model,
            "old_abi_sha256": metadata["abi_sha256"], "new_abi_sha256": new_abi,
            "parameter_words": native.buffers.parameter_words,
            "product_optimizer_blocks": {name: blocks[name] for name in expected},
        }
    require(len(topologies) == 2 and digest(factory) != old_meta["factory_source_sha256"],
            "Expected exactly two repaired Product model topologies and new factory")
    proof = {
        "schema": "generals-product-logical-muon-abi-proof-v1",
        "old_source_asset_sha256": digest(old_cold),
        "old_pool_receipt_sha256": digest(old_input / "pool-migration-receipt.json"),
        "old_optimizer_source_sha256": digest(old_repository / "integrations/spatial_optimizer_layout.py"),
        "new_optimizer_source_sha256": digest(repository / "integrations/spatial_optimizer_layout.py"),
        "old_direct_source_sha256": digest(old_repository / "integrations/direct_spatial_optimization.py"),
        "new_direct_source_sha256": digest(repository / "integrations/direct_spatial_optimization.py"),
        "old_portable_source_sha256": digest(old_repository / "integrations/spatial_policy_bundle.py"),
        "new_portable_source_sha256": digest(repository / "integrations/spatial_policy_bundle.py"),
        "old_factory_source_sha256": old_meta["factory_source_sha256"],
        "new_factory_source_sha256": digest(factory),
        "topologies": topologies,
        "source_policy_sha256": plan["source_product_policy_sha256"],
        "policy_bytes_change": False,
    }
    return proof, directs


def migrate_one(old_asset: Path, old_bundle: Path, new_asset_dir: Path,
                new_bundle_dir: Path, proof: dict, proof_sha: str, factory: Path,
                views: np.ndarray, legal_masks: np.ndarray,
                old_portable_class, new_direct: DirectSpatial) -> dict:
    old_asset_sha = digest(old_asset)
    old = load_asset(old_asset, manifest_sha256=old_asset_sha)
    metadata = old.metadata
    topology = proof["topologies"].get(metadata["model_sha256"])
    require(topology is not None, "Prior asset topology was not proven")
    require(metadata["factory_source_sha256"] == proof["old_factory_source_sha256"] and
            metadata["abi_sha256"] == topology["old_abi_sha256"] and
            metadata["fabric"] == topology["fabric"] and
            old.learner is None, "Prior asset identity or optimizer state differs")
    require(digest(old_bundle / "policy.bin") == metadata["policy_sha256"] and
            read(old_bundle / "asset.json")["abi_sha256"] == topology["old_abi_sha256"],
            "Prior portable bundle differs")
    old_portable = old_portable_class(old_bundle)
    parameters = np.frombuffer(old.policy, "<f4")
    require(np.array_equal(parameters[new_direct.product_action_kernel],
                           np.zeros((8, 8), np.float32)),
            "Frozen or cold Product head is not zero; activation scale would alter logits")
    ancestry = dict(metadata["provenance"]["ancestors"])
    ancestry["prior_native_asset_manifest"] = old_asset_sha
    ancestry["prior_portable_bundle_manifest"] = digest(old_bundle / "spatial-policy.json")
    provenance = dict(metadata["provenance"], operation="source_cleanup",
                      reinforcement_learning_steps_added=0,
                      abi_proof_sha256=proof_sha, ancestors=ancestry)
    new_manifest = write_asset(
        new_asset_dir, fabric=metadata["fabric"],
        factory_source_sha256=proof["new_factory_source_sha256"],
        model_sha256=topology["new_model_sha256"], abi_sha256=topology["new_abi_sha256"],
        policy=old_asset.with_name("policy.bin"), sampler=metadata["sampler"],
        provenance=provenance, training_seeds=metadata["training_seeds"],
    )
    new = load_asset(new_manifest, manifest_sha256=digest(new_manifest))
    new.verify_target(factory_source_sha256=proof["new_factory_source_sha256"],
                      model_sha256=topology["new_model_sha256"],
                      abi_sha256=topology["new_abi_sha256"])
    require(new.policy == old.policy, "Native checkpoint bytes changed")
    export_bundle(new_manifest, digest(new_manifest), factory, new_bundle_dir)
    new_portable = SpatialPlayerPolicy(new_bundle_dir)
    require(digest(new_bundle_dir / "policy.bin") == metadata["policy_sha256"],
            "Portable checkpoint bytes changed")
    with np.load(old_bundle / "weights.npz", allow_pickle=False) as before, \
         np.load(new_bundle_dir / "weights.npz", allow_pickle=False) as after:
        require(set(before.files) == set(after.files) and
                all(np.array_equal(before[name], after[name]) for name in before.files),
                "Portable tensors changed")
    old_logits, new_logits = old_portable.forward(views), new_portable.forward(views)
    require(np.array_equal(old_logits, new_logits), "Portable logits changed")
    new_direct_logits = np.asarray(new_direct.forward(parameters, views))
    direct_max_abs = float(np.max(np.abs(new_direct_logits - old_logits)))
    require(direct_max_abs <= 2e-5, "Repaired native/direct logits differ from old serving")
    for view, mask, before, after in zip(views, legal_masks, old_logits, new_logits, strict=True):
        sampler = metadata["sampler"]
        kwargs = dict(observations=view,
                      neutral_route_bias=sampler.get("neutral_route_bias", 0),
                      weak_owned_route_penalty=sampler.get("weak_owned_route_penalty", 0),
                      doomed_attack_route_penalty=sampler.get("doomed_attack_route_penalty", 0),
                      route_half_weight=sampler.get("route_half_weight", 0),
                      full_action_temperature=sampler.get("full_action_temperature", 1),
                      log_gap_scale=sampler.get("log_gap_scale", 0))
        p0 = structured_action_probabilities(before, mask, sampler["move_temperature"],
                                             sampler["split_temperature"], **kwargs)
        p1 = structured_action_probabilities(after, mask, sampler["move_temperature"],
                                             sampler["split_temperature"], **kwargs)
        require(np.array_equal(p0, p1), "Serving action probabilities changed")
    return {
        "old_asset_sha256": old_asset_sha,
        "new_asset_sha256": digest(new_manifest),
        "old_bundle_sha256": digest(old_bundle / "spatial-policy.json"),
        "new_bundle_sha256": digest(new_bundle_dir / "spatial-policy.json"),
        "old_policy_sha256": metadata["policy_sha256"],
        "new_policy_sha256": metadata["policy_sha256"],
        "old_model_sha256": metadata["model_sha256"],
        "new_model_sha256": topology["new_model_sha256"],
        "old_abi_sha256": topology["old_abi_sha256"],
        "new_abi_sha256": topology["new_abi_sha256"],
        "weights_equal": True, "logits_bitwise_equal": True,
        "new_direct_old_portable_max_abs": direct_max_abs,
        "acting_probabilities_bitwise_equal": True,
    }


def run(old_input: Path, old_repository: Path, repository: Path, output: Path) -> dict:
    old_input, old_repository, repository = (p.resolve(strict=True) for p in
                                              (old_input, old_repository, repository))
    if output.exists():
        raise FileExistsError(output)
    proof, directs = optimizer_identity(old_input, old_repository, repository)
    output.mkdir(parents=True)
    proof_path = output / "abi-proof.json"
    proof_path.write_bytes(canonical_json(proof) + b"\n")
    proof_sha = digest(proof_path)
    views, masks = public_views()
    factory = repository / "integrations/generals_fabric.py"
    old_portable = archived_module("archived_product_portable",
                                   old_repository / "integrations/spatial_policy_bundle.py")
    old_meta = read(old_input / "assets/cold/asset.json")
    source = migrate_one(old_input / "assets/cold/asset.json",
                         old_input / "bundles/cold", output / "assets/cold",
                         output / "bundles/cold", proof, proof_sha, factory, views, masks,
                         old_portable.SpatialPlayerPolicy, directs[old_meta["model_sha256"]])
    frozen = []
    for slot in range(10):
        old_bundle = old_input / "bundles/frozen" / str(slot)
        frozen_meta = read(old_bundle / "asset.json")
        row = migrate_one(old_bundle / "asset.json", old_bundle,
                          output / "assets/frozen" / str(slot),
                          output / "bundles/frozen" / str(slot), proof, proof_sha,
                          factory, views, masks, old_portable.SpatialPlayerPolicy,
                          directs[frozen_meta["model_sha256"]])
        frozen.append(dict(slot=slot, **row))
    require(source["new_policy_sha256"] == frozen[0]["new_policy_sha256"],
            "Frozen slot 0 differs from cold source")
    result = {
        "schema": "generals-product-logical-muon-migration-v1",
        "abi_proof_sha256": proof_sha,
        "source": source,
        "frozen_opponents": frozen,
        "topologies": proof["topologies"],
        "policy_bytes_changed": False,
        "portable_tensors_changed": False,
        "serving_logits_or_acting_changed": False,
    }
    (output / "migration-receipt.json").write_bytes(canonical_json(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-input", "old-repository", "repository", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.old_input, args.old_repository, args.repository, args.output)
    print(json.dumps({"source_policy_sha256": result["source"]["new_policy_sha256"],
                      "topologies": len(result["topologies"]),
                      "frozen_opponents": len(result["frozen_opponents"]),
                      "receipt_sha256": digest(args.output / "migration-receipt.json")}, indent=2))


if __name__ == "__main__":
    main()

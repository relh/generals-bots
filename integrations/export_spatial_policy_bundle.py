"""Export the current native weight asset as an immutable portable policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from integrations.native_spatial_asset import abi_digest, canonical_json, load_asset, write_asset


@lru_cache(maxsize=2)
def realized_model(configuration, factory_source, source_sha):
    config = json.loads(configuration)
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from metta_training.model_config import FabricConfig
    from metta_training.native_build import fabric_fingerprint
    from metta_training.native_fabric import NativeFabricPolicy

    from integrations.direct_spatial_optimization import DirectSpatial
    from integrations.spatial_native_contract import verify_configuration

    verify_configuration(json.dumps(config))
    import jax

    # Asset publication only needs host topology and parameter gather maps.
    with jax.default_device(jax.devices("cpu")[0]):
        native = NativeFabricPolicy(json.dumps(config))
        model = DirectSpatial(native)
    return native, model, fabric_fingerprint(FabricConfig.model_validate(config)), abi_digest(native)


def export_bundle(asset_manifest, manifest_sha256, factory_source, output):
    asset = load_asset(asset_manifest, manifest_sha256=manifest_sha256)
    source_sha = hashlib.sha256(factory_source.read_bytes()).hexdigest()
    config = asset.metadata["fabric"]
    native, model, model_sha, abi_sha = realized_model(canonical_json(config).decode(), factory_source, source_sha)
    asset.verify_target(factory_source_sha256=source_sha, model_sha256=model_sha, abi_sha256=abi_sha)
    parameters = np.frombuffer(asset.policy, "<f4")
    if parameters.size != native.buffers.parameter_words:
        raise ValueError("Native asset parameter allocation differs")
    weights = {}
    for name in ("input_kernel", "context_kernel", "action_kernel",
                 "product_local_kernel", "product_global_kernel", "product_action_kernel"):
        indices = getattr(model, name)
        weights[name] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    for name in ("local_weight", "local_bias", "context_weight", "context_bias"):
        weights[name] = parameters[getattr(model, name)[0]]
    for name in ("global_weight", "global_bias", "global_kernel", "readout_kernel", "output_weight", "output_bias"):
        weights[name] = parameters[getattr(model, name)]
    for i, (source, indices) in enumerate(model.priors):
        weights[f"prior_source_{i}"] = source
        weights[f"prior_weight_{i}"] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    provenance = dict(asset.metadata["provenance"])
    provenance["ancestors"] = dict(provenance["ancestors"], native_asset_manifest=manifest_sha256)
    write_asset(
        output,
        fabric=config,
        factory_source_sha256=source_sha,
        model_sha256=asset.metadata["model_sha256"],
        abi_sha256=asset.metadata["abi_sha256"],
        policy=asset_manifest.with_name("policy.bin"),
        sampler=asset.metadata["sampler"],
        provenance=provenance,
        training_seeds=asset.metadata["training_seeds"],
    )
    np.savez_compressed(output / "weights.npz", **weights)
    files = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("asset.json", "policy.bin", "weights.npz")
    }
    (output / "spatial-policy.json").write_bytes(
        canonical_json(dict(schema="generals-spatial-policy-v1", files=files)) + b"\n"
    )
    return dict(output=str(output), policy_sha256=asset.metadata["policy_sha256"], files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--factory-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.asset, args.manifest_sha256, args.factory_source, args.output)))


if __name__ == "__main__":
    main()

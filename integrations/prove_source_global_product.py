"""Prove Fabric, direct JAX, and portable NumPy agree for the Product policy."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.native_fabric import NativeFabricPolicy

from integrations.direct_spatial_optimization import DirectSpatial
from integrations.native_spatial_asset import canonical_json, load_asset, sha256
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def prove(asset_path: Path, manifest_sha256: str, bundle_path: Path):
    asset = load_asset(asset_path, manifest_sha256=manifest_sha256)
    native = NativeFabricPolicy(json.dumps(asset.metadata["fabric"]))
    direct = DirectSpatial(native)
    portable = SpatialPlayerPolicy(bundle_path)
    parameters = np.frombuffer(asset.policy, "<f4").copy()
    if not np.array_equal(parameters[direct.product_action_kernel], np.zeros((8, 8), np.float32)):
        raise ValueError("Initial residual head differs from zero")
    observations = np.random.default_rng(17290391).normal(0, 0.2, (1, 7056)).astype(np.float32)
    source_direct = np.asarray(direct.forward(parameters, observations))
    source_portable = portable.forward(observations)
    modified = parameters.copy()
    modified[direct.product_action_kernel] = np.random.default_rng(491721).normal(0, .005, (8, 8))
    direct_logits = np.asarray(direct.forward(modified, observations))
    weights = dict(portable.weights, product_action_kernel=modified[direct.product_action_kernel])
    portable_logits = portable._forward(observations, np, weights=weights)
    state = jnp.asarray(native.buffers.pack_state(native.buffers.template))
    obs = jnp.asarray(observations.reshape(1, 7056, 1, 1))
    fn = native.buffers.fn

    def fabric_logits(values):
        sigma = native.buffers.unpack_device(values, state)
        _, prediction = fn.forward(sigma, {"observations": obs})
        return native.model_predictions_device(fn, prediction).reshape(1, 3530)

    fabric_output = np.asarray(fabric_logits(jnp.asarray(modified)))
    cotangent = np.random.default_rng(762913).normal(0, .01, (1, 3530)).astype(np.float32)
    direct_gradient = np.asarray(direct.gradient(jnp.asarray(modified), jnp.asarray(observations),
                                                  jnp.asarray(cotangent)))
    fabric_gradient = np.asarray(jax.grad(lambda values: jnp.sum(
        fabric_logits(values) * jnp.asarray(cotangent)
    ))(jnp.asarray(modified)))
    report = {
        "schema": "generals-source-global-product-parity-v1",
        "asset_manifest_sha256": manifest_sha256,
        "policy_sha256": asset.metadata["policy_sha256"],
        "bundle_manifest_sha256": sha256((bundle_path / "spatial-policy.json").read_bytes()),
        "factory_source_sha256": asset.metadata["factory_source_sha256"],
        "parameter_words": len(parameters),
        "observations_seed": 17290391,
        "perturbation_seed": 491721,
        "cotangent_seed": 762913,
        "fabric_jax_version": jax.__version__,
        "source_direct_portable_max_abs": float(np.max(np.abs(source_direct - source_portable))),
        "nonzero_direct_portable_max_abs": float(np.max(np.abs(direct_logits - portable_logits))),
        "nonzero_fabric_direct_max_abs": float(np.max(np.abs(fabric_output - direct_logits))),
        "nonzero_effect_max_abs": float(np.max(np.abs(direct_logits - source_direct))),
        "fabric_direct_gradient_max_abs": float(np.max(np.abs(fabric_gradient - direct_gradient))),
        "fabric_direct_head_gradient_max_abs": float(np.max(np.abs(
            fabric_gradient[direct.product_action_kernel] - direct_gradient[direct.product_action_kernel]
        ))),
        "gradient_norm": float(np.linalg.norm(direct_gradient)),
    }
    if (report["source_direct_portable_max_abs"] > 2e-5
            or report["nonzero_direct_portable_max_abs"] > 2e-5
            or report["nonzero_fabric_direct_max_abs"] > 2e-5
            or report["fabric_direct_gradient_max_abs"] > 3e-6
            or report["nonzero_effect_max_abs"] < 1e-5):
        raise ValueError(f"Product policy parity failed: {report}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prove(args.asset, args.manifest_sha256, args.bundle)
    args.output.write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(dict(result, receipt_sha256=sha256(args.output.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()

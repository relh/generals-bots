"""Bounded CPU proof of the actual cross-to-radius-two checkpoint mapping."""

import argparse
import copy
import hashlib
import importlib.util
import json
import signal
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-bundle", "factory-source", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(600)
    if jax.default_backend() != "cpu":
        raise ValueError("This readiness audit must run on CPU before GPU scheduling")
    args.output.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", args.factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from metta_training.native_fabric import NativeFabricPolicy

    from integrations.direct_spatial_optimization import DirectSpatial
    from integrations.memoryless_optimization import verify_configuration
    from integrations.spatial_context_transfer import extend_flat, parameter_mapping, qualified_mapping
    from integrations.spatial_policy_bundle import SpatialPlayerPolicy

    portable = SpatialPlayerPolicy(args.parent_bundle)
    config = json.loads((args.parent_bundle / "build.json").read_text())["config"]["fabric"]
    config["platform"] = "cpu"
    assert config["options"]["context_radius"] == 1.01
    verify_configuration(json.dumps(config))
    old_policy = NativeFabricPolicy(json.dumps(config))
    old = DirectSpatial(old_policy)
    wide_config = copy.deepcopy(config)
    wide_config["options"]["context_radius"] = 2.01
    verify_configuration(json.dumps(wide_config))
    new_policy = NativeFabricPolicy(json.dumps(wide_config))
    new = DirectSpatial(new_policy)
    mapping = parameter_mapping(old, new, old_policy.buffers, new_policy.buffers)
    np.testing.assert_array_equal(mapping, qualified_mapping())
    parameters = np.frombuffer((args.parent_bundle / "policy.bin").read_bytes(), "<f4")
    extended = extend_flat(parameters, mapping)
    np.save(args.output / "parameter-mapping.npy", mapping)
    (args.output / "extended-policy.bin").write_bytes(extended.tobytes())
    np.savez(args.output / "context-layout.npz", old=old.context_kernel, new=new.context_kernel)
    print("ACTUAL_CONTEXT_MAPPING_OK", parameters.size, extended.size, flush=True)

    source = np.load(args.views, allow_pickle=False)
    views = np.asarray(source["views"], np.float32)
    assert views.shape[1] == portable.observation_size
    before = np.asarray(old.forward(jnp.asarray(parameters), jnp.asarray(views)))
    after = np.asarray(new.forward(jnp.asarray(extended), jnp.asarray(views)))
    np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-5)
    serving = copy.copy(portable)
    serving.weights = {k: v.copy() for k, v in portable.weights.items()}
    serving.weights["context_kernel"] = np.pad(portable.weights["context_kernel"],
                                               ((1, 1), (1, 1), (0, 0), (0, 0)))
    np.testing.assert_allclose(serving.forward(views), portable.forward(views), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(serving.forward(views), after, rtol=2e-5, atol=2e-5)
    print("ZERO_EXTENSION_FORWARD_OK", float(np.max(np.abs(after - before))), flush=True)

    # Compare against the real Fabric execution/backward, not another copy of
    # the convolution implementation. Perturb newly added weights as well, so
    # a zero-only test cannot accidentally conceal unsupported new connections.
    new_policy.set_training_step(0)
    trial = extended.copy()
    introduced = np.unique(new.context_kernel[new.context_kernel >= 0])
    introduced = introduced[mapping[introduced] < 0]
    trial[introduced] = np.random.default_rng(853).normal(0, .01, len(introduced)).astype(np.float32)
    batch = 2
    obs = jnp.asarray(views[:batch, None, :])
    state = jnp.asarray(np.repeat(new_policy.buffers.pack_state(new_policy.buffers.template), batch, axis=0))
    native, _, tape = new_policy._forward_arrays(jnp.asarray(trial), state, obs,
                                                jnp.ones((batch, 1)), batch, 1, False)
    direct = new.forward(jnp.asarray(trial), obs)
    np.testing.assert_allclose(direct, native, rtol=2e-5, atol=2e-5)
    cot = jnp.asarray(np.random.default_rng(854).normal(0, .01, native.shape).astype(np.float32))
    native_grad = np.asarray(new_policy.backward_device_arrays(tape, cot[..., :-1], cot[..., -1]))
    direct_grad = np.asarray(new.gradient(jnp.asarray(trial), obs, cot))
    np.testing.assert_allclose(direct_grad, native_grad, rtol=2e-4, atol=2e-5)
    assert np.isfinite(direct_grad).all() and np.linalg.norm(direct_grad[introduced]) > 0
    result = dict(scope="CPU layout, checkpoint transfer, portable forward and native gradient; no SPS/strength claim",
                  parent_checkpoint_sha256=hashlib.sha256(parameters.tobytes()).hexdigest(),
                  extended_checkpoint_sha256=hashlib.sha256(extended.tobytes()).hexdigest(),
                  original_parameter_words=int(parameters.size), extended_parameter_words=int(extended.size),
                  new_parameters=int(len(introduced)), views=len(views),
                  zero_extension_max_error=float(np.max(np.abs(after - before))),
                  native_forward_max_error=float(np.max(np.abs(np.asarray(direct) - np.asarray(native)))),
                  native_gradient_max_error=float(np.max(np.abs(native_grad - direct_grad))),
                  new_neighbor_gradient_norm=float(np.linalg.norm(direct_grad[introduced])))
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

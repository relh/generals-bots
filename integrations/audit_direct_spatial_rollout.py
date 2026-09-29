"""Compare sequential original and direct rollout with real views and resets."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "factory-source", "checkpoint", "views", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--smoke-cpu", action="store_true")
    args = parser.parse_args()
    if args.batch < 2 or args.ticks < 3:
        raise ValueError("Require multiple actors and sequential ticks")
    if not args.smoke_cpu and jax.devices()[0].platform != "gpu":
        raise RuntimeError("Rollout GPU parity requires GPU execution")
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", args.factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from integrations.direct_spatial_optimization import install
    from metta_training.native_fabric import NativeFabricPolicy

    os.environ["METTA_DIRECT_SPATIAL_ROLLOUT"] = "1"
    original, _ = install()
    policy = NativeFabricPolicy(json.dumps(json.loads(args.build.read_text())["config"]["fabric"]))
    raw = args.checkpoint.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Checkpoint checksum mismatch")
    weights = np.frombuffer(raw, np.float32)
    assert weights.size == policy.buffers.parameter_words and np.isfinite(weights).all()
    parameters = jnp.asarray(weights)
    views = np.fromfile(args.views, np.float32).reshape(-1, 4851)
    assert len(views) >= 2 and np.isfinite(views).all()
    state = jnp.asarray(np.repeat(policy.buffers.pack_state(policy.buffers.template), args.batch, axis=0))
    direct_state = state
    maximum_error = 0.0
    for tick in range(args.ticks):
        indices = (np.arange(args.batch) + args.batch * tick) % len(views)
        observations = jnp.asarray(views[indices, None, :])
        # Different reset periods across seats and varying, real input views.
        terminals = jnp.asarray((tick % (np.arange(args.batch) % 5 + 2) == 0)[:, None], jnp.float32)
        expected, state, _ = original(policy, parameters, state, observations, terminals, args.batch, 1, True)
        actual, direct_state, _ = policy._forward_arrays(
            parameters, direct_state, observations, terminals, args.batch, 1, True,
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        maximum_error = max(maximum_error, float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
    result = dict(scope="Sequential rollout equivalence; not training throughput or arena strength",
                  platform=jax.devices()[0].platform, batch=args.batch, ticks=args.ticks,
                  checkpoint_sha256=args.sha256, max_absolute_error=maximum_error)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

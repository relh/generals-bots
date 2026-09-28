"""Compare optimization rows with original sequence outputs and PPO cotangents."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--factory-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Defaults to the native initializer")
    parser.add_argument("--sha256", help="Required with --checkpoint")
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--time", type=int, default=3)
    args = parser.parse_args()
    assert args.batch >= 2 and args.time >= 3
    assert (args.checkpoint is None) == (args.sha256 is None)
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", args.factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from integrations.memoryless_optimization import install
    from metta_training.native_fabric import NativeFabricPolicy

    original_forward, original_backward = install()
    build = json.loads(args.build.read_text())
    policy = NativeFabricPolicy(json.dumps(build["config"]["fabric"]))
    raw = args.checkpoint.read_bytes() if args.checkpoint else policy.initial_parameters
    parameter_sha256 = hashlib.sha256(raw).hexdigest()
    if args.checkpoint:
        assert parameter_sha256 == args.sha256
    weights = np.frombuffer(raw, np.float32)
    assert weights.size == policy.buffers.parameter_words and np.isfinite(weights).all()
    parameters = jnp.asarray(weights)
    views = np.fromfile(args.views, np.float32).reshape(-1, policy.observation_size)
    assert len(views) >= args.batch * args.time and np.isfinite(views).all()
    observations = jnp.asarray(views[: args.batch * args.time].reshape(args.batch, args.time, -1))
    state = jnp.asarray(np.repeat(policy.buffers.pack_state(policy.buffers.template), args.batch, axis=0))
    # Real incoming state plus selective resets exercise the claimed independence.
    _, state, _ = original_forward(
        policy, parameters, state, observations[:, :1],
        jnp.zeros((args.batch, 1), jnp.float32), args.batch, 1, True,
    )
    done = np.zeros((args.batch, args.time), np.float32)
    done[0, 1] = 1
    done[1, -1] = 1
    done = jnp.asarray(done)
    expected, _, original_tape = original_forward(
        policy, parameters, state, observations, done, args.batch, args.time, False,
    )
    actual, _, rows_tape = policy._forward_arrays(
        parameters, state, observations, done, args.batch, args.time, False,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    rng = np.random.default_rng(17)
    actor = jnp.asarray(rng.normal(0, 0.01, (*expected.shape[:2], policy.output_size - 1)).astype(np.float32))
    critic = jnp.asarray(rng.normal(0, 0.1, expected.shape[:2]).astype(np.float32))
    expected_gradient = np.asarray(original_backward(policy, original_tape, actor, critic))
    actual_gradient = np.asarray(policy.backward_device_arrays(rows_tape, actor, critic))
    assert np.isfinite(actual_gradient).all() and np.isfinite(expected_gradient).all()
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-4)
    # Rollout path is unchanged, including its carried state.
    expected_rollout = original_forward(policy, parameters, state, observations[:, :1], done[:, :1], args.batch, 1, True)
    actual_rollout = policy._forward_arrays(parameters, state, observations[:, :1], done[:, :1], args.batch, 1, True)
    np.testing.assert_array_equal(actual_rollout[0], expected_rollout[0])
    np.testing.assert_array_equal(actual_rollout[1], expected_rollout[1])
    result = dict(
        scope="Numerical optimization audit on public views; not arena strength or training throughput",
        batch=args.batch, time=args.time, partial_resets=True, nonempty_incoming_state=True,
        checkpoint_sha256=args.sha256, parameter_words=weights.size,
        parameter_source="checkpoint" if args.checkpoint else "native_initializer",
        parameter_sha256=parameter_sha256,
        max_output_difference=float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))),
        max_parameter_gradient_difference=float(np.max(np.abs(actual_gradient - expected_gradient))),
        original_gradient_norm=float(np.linalg.norm(expected_gradient)),
        exact_rollout_outputs_and_state=True,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

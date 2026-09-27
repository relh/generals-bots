"""Verify that removing unused teacher labels preserves transitions and PPO gradients."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.native_fabric import NativeFabricPolicy

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-build", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    original, target = [json.loads(path.read_text())["config"] for path in (args.original_build, args.build)]
    assert target["fabric"]["teacher"] is None
    for phase in original["fabric"]["teacher"]["phases"]:
        assert (phase["coefficient"], phase["ppo_coefficient"], phase["action_mix"]) == (0, 1, 0)
    options = original["python_environment"]["options"] | {"parallel_games": 32, "coworld_pool_size": 16}
    context = EnvironmentContext(seed=1334, index=0, mode="train", output=args.output.parent)
    labeled = BatchedGeneralsPufferEnvironment(context=context, **options)
    plain = BatchedGeneralsPufferEnvironment(context=context, **(options | {"supervise_teacher": False}))
    frames = [[], []]
    try:
        left, right = labeled.reset_device("1334"), plain.reset_device("1334")
        for step in range(64):
            np.testing.assert_array_equal(np.asarray(left[0])[:, :6174], np.asarray(right[0]))
            np.testing.assert_array_equal(np.asarray(left[1]), np.asarray(right[1]))
            if step < 3:
                frames[0].append(np.asarray(left[0])[:2])
                frames[1].append(np.asarray(right[0])[:2])
            moves = np.argmax(np.asarray(left[1])[:, :1765], axis=1)
            actions = jnp.asarray(np.stack((moves, np.full(32, step % 2)), axis=1), dtype=jnp.float32)
            left, right = labeled.step_device(actions), plain.step_device(actions)
            for a, b in zip(left[1:], right[1:], strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree.leaves(labeled.states), jax.tree.leaves(plain.states), strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            np.testing.assert_array_equal(np.asarray(labeled.keys), np.asarray(plain.keys))
    finally:
        labeled.close()
        plain.close()
    policies = [NativeFabricPolicy(json.dumps(config["fabric"])) for config in (original, target)]
    parameters = [policy.initialize(1334) for policy in policies]
    assert parameters[0] == parameters[1]
    assert policies[0].state_words == policies[1].state_words
    predictions, gradients = [], []
    rng = np.random.default_rng(1334)
    actor = jnp.asarray(rng.normal(size=(2, 3, 1767)).astype(np.float32) / 100)
    critic = jnp.asarray(rng.normal(size=(2, 3)).astype(np.float32) / 100)
    for policy, weights, rows in zip(policies, parameters, frames, strict=True):
        state = np.zeros((2, policy.state_words), np.float32)
        observations = np.stack(rows, axis=1).astype(np.float32)
        terminals = np.zeros((2, 3), np.float32)
        prediction, _, tape = policy.forward(
            weights, state.tobytes(), observations.tobytes(), terminals.tobytes(), 2, 3, False
        )
        predictions.append(np.frombuffer(prediction, np.float32))
        gradients.append(np.asarray(policy.backward_device_arrays(tape, actor, critic)))
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_allclose(gradients[0], gradients[1], rtol=2e-5, atol=2e-5)
    result = dict(
        transition_steps=64 * 32, identical_initial_parameters=True, identical_predictions=True,
        max_gradient_difference=float(np.max(np.abs(gradients[0] - gradients[1]))),
        source_transport=frames[0][0].shape[-1], target_transport=frames[1][0].shape[-1],
        scope="GPU transition and full-graph cotangent parity; no checkpoint transfer or strength claim",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

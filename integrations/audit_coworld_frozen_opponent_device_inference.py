"""Prove a frozen Fabric checkpoint can act on device without host action transfer."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig
from metta_training.native_fabric import NativeFabricPolicy, compile_policy

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def served_logits(policy: FrozenPolicy, values: np.ndarray) -> np.ndarray:
    batch = values.shape[0]
    transported = np.zeros((batch, policy.policy.input_size), np.float32)
    transported[:, : values.shape[1]] = values
    with jax.default_device(policy.device):
        outputs, _, _ = policy.policy.forward(
            policy.parameters,
            bytes(batch * policy.policy.state_words * 4),
            transported.tobytes(),
            bytes(batch * 4),
            batch, 1, True,
        )
    return np.frombuffer(outputs, np.float32).reshape(batch, -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    manifest = json.loads(args.build.read_text())
    assert manifest["model_state_words"] == 22956
    fabric = manifest["config"]["fabric"]
    assert fabric["observation_size"] == 14 * 21 * 21
    assert fabric["action_sizes"] == [1765, 2]
    assert fabric["compiler"] == "standard"
    native = NativeFabricPolicy(json.dumps(fabric))
    batch = 16
    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(manifest["config"]["python_environment"]["options"])
    options.update(parallel_games=batch, coworld_pool_size=batch,
                   supervise_teacher=False, deduplicate_opponent_branches=False)
    environment = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1391, index=0, mode="evaluate", output=args.output),
        **options,
    )
    try:
        observation = environment.reset("1391:0:0")
        values = np.asarray(observation.values, np.float32)
        masks = np.asarray(observation.action_masks, bool)
        assert values.shape == (batch, native.observation_size)
        served = FrozenPolicy(FrozenPolicyConfig(
            build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0"
        ))
        served.reset("1391:0:0")
        expected = served_logits(served, values)
        with jax.default_device(jax.devices("gpu")[0]):
            parameters = jnp.asarray(np.frombuffer(args.checkpoint.read_bytes(), np.float32).copy())
            state = jnp.zeros((batch, native.state_words), jnp.float32)
            function = compile_policy(
                native.graph, inputs=native.inputs, outputs=native.outputs,
                slots=batch, output_order=native.output_order, standard=native.standard_compile,
            )
            sigma = native.buffers.unpack_device(parameters, state)
            board = jnp.asarray(values).reshape(batch, 1, native.observation_size)
            board = board.transpose(1, 2, 0)[..., None]
            board = jnp.pad(board, ((0, 0), (0, native.graph_input_size - native.observation_size),
                                    (0, 0), (0, 0)))
            forward = jax.jit(lambda state, obs: function.forward(
                state, {"observations": obs}, reset=jnp.zeros((1, batch), bool)
            ))
            advanced, prediction = forward(sigma, board)
            actual = np.asarray(native.model_predictions_device(function, prediction))[:, 0]
            _, continued = forward(advanced, board)
            _, restarted = forward(sigma, board)
            assert np.array_equal(
                np.asarray(native.model_predictions_device(function, continued)),
                np.asarray(native.model_predictions_device(function, restarted)),
            ), "Frozen model reads carried state; preserve it between steps"
        assert actual.shape == expected.shape
        max_absolute_logit_error = float(np.max(np.abs(actual - expected)))
        print(json.dumps({
            "max_absolute_logit_error": max_absolute_logit_error,
            "mean_absolute_logit_error": float(np.mean(np.abs(actual - expected))),
            "unequal_logit_words": int(np.count_nonzero(actual != expected)),
        }), flush=True)
        offset = 0
        action_mismatches = []
        for size in native.action_sizes:
            legal = masks[:, offset:offset + size]
            assert legal.any(axis=1).all()
            action_mismatches.append(int(np.count_nonzero(
                np.argmax(np.where(legal, actual[:, offset:offset + size], -np.inf), axis=1)
                != np.argmax(np.where(legal, expected[:, offset:offset + size], -np.inf), axis=1)
            )))
            offset += size
        print(json.dumps({"action_mismatches": action_mismatches}), flush=True)
        assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4)
        assert action_mismatches == [0] * len(native.action_sizes)
        result = {
            "checkpoint_sha256": args.sha256,
            "model_sha256": manifest["model_sha256"],
            "batch": batch,
            "logits_shape": list(actual.shape),
            "logits_bit_equal": bool(np.array_equal(actual, expected)),
            "max_absolute_logit_error": max_absolute_logit_error,
            "masked_actions_equal": True,
            "state_words": native.state_words,
            "carried_state_changes_predictions": False,
            "host_action_transfer_in_device_forward": False,
        }
        (args.output / "inference.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()

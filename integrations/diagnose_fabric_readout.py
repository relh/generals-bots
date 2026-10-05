"""Check whether a frozen Fabric actor's global heads depend on its input."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from metta_training.native_fabric import NativeFabricPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.build.read_text())
    model = manifest["config"]["fabric"]
    with jax.default_device(jax.devices("cuda")[0]):
        policy = NativeFabricPolicy(json.dumps(model))
        initialized = policy.initialize(7)
        trained = args.checkpoint.read_bytes() if args.checkpoint else None
        assert trained is None or len(trained) == len(initialized)
        cells = model["options"]["height"] * model["options"]["width"]
        inputs = np.zeros((2, policy.input_size), np.float32)
        if args.real:
            import jax.numpy as jnp
            from generals.core import game
            from integrations.puffer_codec import encode_observation

            board = jnp.zeros((21, 21), dtype=jnp.int32).at[2, 2].set(1).at[18, 18].set(2)
            state = game.create_initial_state(board)
            later = state._replace(armies=state.armies.at[2, 2].set(20), time=jnp.int32(200))
            for i, current in enumerate((state, later)):
                values, _ = encode_observation(game.get_observation(current, 0), factorized_actions=True)
                inputs[i, :policy.observation_size] = np.asarray(values)
        else:
            inputs[1, :policy.observation_size] = np.random.default_rng(7).uniform(
                -0.25, 0.25, policy.observation_size
            )
        for label, parameters in (("initialized", initialized), ("trained", trained)):
            if parameters is None:
                continue
            outputs = []
            for values in inputs:
                raw, _, _ = policy.forward(parameters, bytes(policy.state_words * 4),
                                           values.tobytes(), bytes(4), 1, 1, True)
                outputs.append(np.frombuffer(raw, np.float32))
            outputs = np.stack(outputs)
            difference = outputs[1] - outputs[0]
            print(json.dumps({
                "weights": label,
                "checkpoint_sha256": hashlib.sha256(parameters).hexdigest(),
                "real_inputs": args.real,
                "input_max_delta": float(np.max(np.abs(inputs[1] - inputs[0]))),
                "move_logit_max_delta": float(np.max(np.abs(difference[:4 * cells]))),
                "pass_logit_delta": float(difference[4 * cells]),
                "split_logit_deltas": difference[4 * cells + 1:4 * cells + 3].tolist(),
                "value_delta": float(difference[-1]),
            }), flush=True)


if __name__ == "__main__":
    main()

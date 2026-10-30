"""Check that a saved flat actor acts legally in the one-seat device environment."""

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

from integrations.metta_puffer import BatchedGeneralsFrozenOpponentPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices("gpu")
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    manifest = json.loads(args.build.read_text())
    assert manifest["model_sha256"] == "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
    options = dict(manifest["config"]["python_environment"]["options"])
    assert options["factorized_actions"] is False
    options.update(
        parallel_games=16, coworld_pool_size=16, horizon=8,
        frozen_build=str(args.build), frozen_checkpoint=str(args.checkpoint),
        frozen_sha256=args.sha256,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    env = BatchedGeneralsFrozenOpponentPufferEnvironment(
        context=EnvironmentContext(seed=1386, index=0, mode="train", output=args.output),
        **options,
    )
    try:
        learner_values, learner_masks = env.reset_device("1386:0:0")
        values, masks = env._cached_values, env._cached_masks
        opponent_sides = 1 - env.sides
        opponent_values = values[env._rows, opponent_sides]
        opponent_masks = masks[env._rows, opponent_sides]
        actions = np.asarray(env._frozen_actions(opponent_values, opponent_masks))
        legal = np.asarray(opponent_masks, bool)
        assert actions.shape == (16, 1)
        assert np.all(legal[np.arange(16), actions[:, 0]])

        served = FrozenPolicy(FrozenPolicyConfig(
            build=args.build, checkpoint=args.checkpoint,
            sha256=args.sha256, device="cuda:0",
        ))
        model = served.policy
        transported = np.zeros((16, model.input_size), np.float32)
        transported[:, :model.observation_size] = np.asarray(opponent_values)
        with jax.default_device(served.device):
            prediction, _, _ = model.forward(
                served.parameters, bytes(16 * model.state_words * 4),
                transported.tobytes(), bytes(16 * 4), 16, 1, True,
            )
        logits = np.frombuffer(prediction, np.float32).reshape(16, -1)
        expected = np.argmax(np.where(legal, logits[:, :legal.shape[1]], -np.inf), axis=1)
        assert np.array_equal(actions[:, 0], expected), "Frozen opponent disagrees with served actor"

        learner_legal = np.asarray(learner_masks, bool)
        learner_actions = np.argmax(learner_legal, axis=1).astype(np.int32)[:, None]
        step = env.step_device(jnp.asarray(learner_actions))
        assert np.asarray(step[0]).shape == np.asarray(learner_values).shape
        assert np.asarray(step[1]).shape == np.asarray(learner_masks).shape
        result = {
            "model_sha256": manifest["model_sha256"],
            "checkpoint_sha256": args.sha256,
            "games": 16,
            "frozen_actions_legal": True,
            "frozen_actions_match_served": True,
            "one_step_shapes_valid": True,
        }
        (args.output / "audit.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

"""Verify an 11-plane flat learner faces the pinned 14-plane generation-0 actor."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import (
    BatchedGeneralsFrozenOpponentPufferEnvironment,
    BatchedGeneralsPufferEnvironment,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner-build", type=Path, required=True)
    parser.add_argument("--frozen-build", type=Path, required=True)
    parser.add_argument("--frozen-checkpoint", type=Path, required=True)
    parser.add_argument("--legacy-fabric", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices("gpu")
    frozen_sha = "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf"
    assert hashlib.sha256(args.frozen_checkpoint.read_bytes()).hexdigest() == frozen_sha
    learner_build = json.loads(args.learner_build.read_text())
    frozen_build = json.loads(args.frozen_build.read_text())
    assert learner_build["model_sha256"] == "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
    assert frozen_build["model_sha256"] == "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
    learner_options = dict(learner_build["config"]["python_environment"]["options"])
    frozen_options = dict(frozen_build["config"]["python_environment"]["options"])
    assert learner_options["factorized_actions"] is False and learner_options["directional_features"]
    assert frozen_options["factorized_actions"] is True and frozen_options["hint_features"]
    learner_options.update(
        parallel_games=16, coworld_pool_size=16, horizon=8,
        frozen_codec="hinted_gen0", frozen_build=str(args.frozen_build),
        frozen_checkpoint=str(args.frozen_checkpoint), frozen_sha256=frozen_sha,
        frozen_legacy_fabric=str(args.legacy_fabric),
    )
    frozen_options.update(parallel_games=16, coworld_pool_size=16, horizon=8)
    args.output.mkdir(parents=True, exist_ok=False)
    env = BatchedGeneralsFrozenOpponentPufferEnvironment(
        context=EnvironmentContext(seed=1386, index=0, mode="train", output=args.output / "learner"),
        **learner_options,
    )
    source = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1386, index=0, mode="evaluate", output=args.output / "source"),
        **frozen_options,
    )
    try:
        learner_values, learner_masks = env.reset_device("1386:0:0")
        assert np.asarray(learner_values).shape == (16, 11 * 21 * 21)
        assert np.asarray(learner_masks).shape == (16, 8 * 21 * 21 + 1)
        sides = 1 - env.sides
        states = jax.tree.map(lambda field: field[env._frozen_rows], env.states)
        frozen_values, frozen_masks = env._frozen_observe(states, sides)
        reference_values, reference_masks = jax.vmap(source.base._observe)(states, sides)
        np.testing.assert_array_equal(np.asarray(frozen_values), np.asarray(reference_values))
        np.testing.assert_array_equal(np.asarray(frozen_masks), np.asarray(reference_masks))
        assert np.asarray(frozen_values).shape == (16, 14 * 21 * 21)
        assert np.asarray(frozen_masks).shape == (16, 4 * 21 * 21 + 3)

        action = np.asarray(env._frozen_actions(frozen_values, frozen_masks))
        assert action.shape == (16, 1)
        flat_masks = np.asarray(env._cached_masks[env._rows, sides], bool)
        assert flat_masks[np.arange(16), action[:, 0]].all()

        model = env._frozen_model
        transported = np.zeros((16, model.input_size), np.float32)
        transported[:, :model.observation_size] = np.asarray(frozen_values)
        with jax.default_device(jax.devices("gpu")[0]):
            prediction, _, _ = model.forward(
                args.frozen_checkpoint.read_bytes(), bytes(16 * model.state_words * 4),
                transported.tobytes(), bytes(16 * 4), 16, 1, True,
            )
        logits = np.frombuffer(prediction, np.float32).reshape(16, -1)
        legal = np.asarray(frozen_masks, bool)
        cells = 21 * 21
        move = np.argmax(np.where(legal[:, :4 * cells + 1], logits[:, :4 * cells + 1], -np.inf), axis=1)
        split = np.argmax(np.where(legal[:, 4 * cells + 1:4 * cells + 3],
                                   logits[:, 4 * cells + 1:4 * cells + 3], -np.inf), axis=1)
        expected = np.where(move == 4 * cells, 8 * cells, move + 4 * cells * split)
        np.testing.assert_array_equal(action[:, 0], expected)
        learner_action = np.argmax(np.asarray(learner_masks, bool), axis=1).astype(np.int32)[:, None]
        transition = env.step_device(jnp.asarray(learner_action))
        assert np.asarray(transition[0]).shape == np.asarray(learner_values).shape
        assert np.asarray(transition[1]).shape == np.asarray(learner_masks).shape
        side_counts = np.bincount(np.asarray(env.sides), minlength=2).tolist()
        assert side_counts == [8, 8]
        result = dict(games=16, side_counts=side_counts, learner_planes=11, frozen_planes=14,
                      frozen_checkpoint_sha256=frozen_sha, public_observations_equal=True,
                      frozen_actions_match_served=True, frozen_flat_actions_legal=True,
                      one_step_shapes_valid=True)
        (args.output / "audit.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        env.close()
        source.close()


if __name__ == "__main__":
    main()

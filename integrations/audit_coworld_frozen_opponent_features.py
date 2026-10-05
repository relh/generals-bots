"""Verify quarter-scale learner observations restore the frozen actor's original codec."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--target-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    source = json.loads(args.source_build.read_text())
    target = json.loads(args.target_build.read_text())
    assert source["model_sha256"] == "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
    assert target["model_sha256"] == "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
    source_options = dict(source["config"]["python_environment"]["options"])
    target_options = dict(target["config"]["python_environment"]["options"])
    assert target_options.pop("move_hint_scale") == target_options.pop("split_hint_scale") == 0.25
    assert target_options == source_options
    args.output.mkdir(parents=True, exist_ok=False)
    environments = []
    for name, options in (("source", source_options), ("target", target_options | {
        "move_hint_scale": 0.25, "split_hint_scale": 0.25,
    })):
        environment = BatchedGeneralsPufferEnvironment(
            context=EnvironmentContext(seed=1397, index=0, mode="evaluate", output=args.output / name),
            **(options | {
                "parallel_games": 16, "coworld_pool_size": 16,
                "supervise_teacher": False, "deduplicate_opponent_branches": False,
            }),
        )
        environments.append(environment)
    try:
        original, scaled = (environment.reset("1397:0:0") for environment in environments)
        expected = np.asarray(original.values, np.float32)
        calibrated = np.asarray(scaled.values, np.float32)
        assert expected.shape == calibrated.shape == (16, 14 * 21 * 21)
        assert np.array_equal(original.action_masks, scaled.action_masks)
        assert np.array_equal(np.asarray(environments[0].sides), np.asarray(environments[1].sides))
        for left, right in zip(jax.tree.leaves(environments[0].states),
                               jax.tree.leaves(environments[1].states), strict=True):
            assert np.array_equal(np.asarray(left), np.asarray(right))
        restored = calibrated.reshape(16, 14, 21 * 21).copy()
        restored[:, 2:8] *= 4
        restored = restored.reshape(16, -1)
        assert np.array_equal(restored, expected)
        result = {
            "games": 16,
            "source_model_sha256": source["model_sha256"],
            "learner_model_sha256": target["model_sha256"],
            "same_game_states": True,
            "same_legal_masks": True,
            "restored_frozen_features_bit_equal": True,
        }
        (args.output / "features.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        for environment in environments:
            environment.close()


if __name__ == "__main__":
    main()

"""Contract checks for the optional native PufferLib bridge."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("metta_training")

from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import GeneralsPufferEnvironment


def test_fogged_observation_mask_and_finite_episode():
    context = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    env = GeneralsPufferEnvironment(context=context, board_size=6, horizon=12)
    observation = env.reset("73:0:0")
    assert len(observation.values[0]) == env.spec.observation_size
    assert len(observation.action_masks[0]) == env.spec.action_sizes[0]
    assert observation.action_masks[0][-1]
    assert np.isfinite(observation.values[0]).all()

    for step in range(12):
        result = env.step([[env.spec.action_sizes[0] - 1]])
        assert result.episode_done is (step == 11)
        assert result.terminated == [result.episode_done]
        assert np.isfinite(result.rewards).all()
    assert result.score == 0
    assert result.perf == 0.5
    env.close()

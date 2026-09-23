"""Contract checks for the optional native PufferLib bridge."""

from pathlib import Path

import jax
import numpy as np
import pytest

pytest.importorskip("metta_training")

from metta_training.environment import EnvironmentContext

from generals.agents import ExpanderAgent
from generals.core import game
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


@pytest.mark.parametrize("opponent", ["random", "hunter", "harvester", "mixed"])
def test_other_opponents_keep_the_numeric_contract(opponent):
    context = EnvironmentContext(seed=73, index=0, mode="evaluate", output=Path("/tmp"))
    env = GeneralsPufferEnvironment(context=context, board_size=6, horizon=12, opponent=opponent)
    observation = env.reset("73:0:0")
    action = observation.action_masks[0].index(True)
    transition = env.step([[action]])
    assert len(transition.observation.values[0]) == env.spec.observation_size
    assert len(transition.observation.action_masks[0]) == env.spec.action_sizes[0]
    assert np.isfinite(transition.rewards).all()
    env.close()


def test_optional_teacher_reward_uses_public_action_and_keeps_terminal_score():
    context = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    coached = GeneralsPufferEnvironment(
        context=context,
        board_size=6,
        horizon=30,
        opponent="random",
        shaping_weight=0,
        teacher="expander",
        imitation_weight=0.4,
    )
    baseline = GeneralsPufferEnvironment(context=context, board_size=6, horizon=30, opponent="random", shaping_weight=0)
    coached.reset("73:0:0")
    baseline.reset("73:0:0")
    pass_index = coached.spec.action_sizes[0] - 1
    teacher = ExpanderAgent()
    for _ in range(20):
        observation = game.get_observation(coached.state, coached.side)
        opponent_key, _ = jax.random.split(coached.key)
        suggestion = np.asarray(teacher.act(observation, jax.random.fold_in(opponent_key, 37)))
        if suggestion[0] == 0:
            break
        assert not coached.step([[pass_index]]).episode_done
        assert not baseline.step([[pass_index]]).episode_done
    else:
        pytest.fail("Teacher never suggested a move")
    board_cells = coached.size**2
    action_index = int(
        suggestion[4] * 4 * board_cells + suggestion[3] * board_cells + suggestion[1] * coached.size + suggestion[2]
    )
    rewarded = coached.step([[action_index]])
    unshaped = baseline.step([[action_index]])
    assert rewarded.rewards[0] - unshaped.rewards[0] == pytest.approx(0.4)
    assert rewarded.score == unshaped.score


def test_supervised_teacher_provides_legal_targets_only_during_training():
    training = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    evaluating = training.model_copy(update={"mode": "evaluate"})
    train_env = GeneralsPufferEnvironment(
        context=training, board_size=6, horizon=12, teacher="harvester", supervise_teacher=True
    )
    eval_env = GeneralsPufferEnvironment(
        context=evaluating, board_size=6, horizon=12, teacher="harvester", supervise_teacher=True
    )
    for env, expected_weight in ((train_env, 1.0), (eval_env, 0.0)):
        observation = env.reset("73:0:0")
        assert env.spec.teacher
        for _ in range(3):
            target = observation.teachers[0]
            assert target.weights == [expected_weight]
            assert sum(target.probabilities) == expected_weight
            assert all(
                not probability or legal
                for probability, legal in zip(target.probabilities, observation.action_masks[0])
            )
            observation = env.step([[env.spec.action_sizes[0] - 1]]).observation
        env.close()

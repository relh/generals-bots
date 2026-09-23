"""Contract checks for the optional native PufferLib bridge."""

from pathlib import Path

import jax
import numpy as np
import pytest

pytest.importorskip("metta_training")

from metta_training.environment import EnvironmentContext, NativeEnvironment

from generals.agents import ExpanderAgent
from generals.core import game
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment, GeneralsPufferEnvironment
from integrations.puffer_codec import decode_action


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


def test_classic10_evaluation_maps_match_classic_scenario_parameters():
    context = EnvironmentContext(seed=901, index=0, mode="evaluate", output=Path("/tmp"))
    env = GeneralsPufferEnvironment(
        context=context, board_size=10, horizon=800, opponent="mixed", classic_maps=True
    )
    assert env.env._fixed_dims == (10, 10)
    assert env.env.num_players == 2
    assert env.env.truncation == 800
    assert env.env.min_generals_distance == 8
    assert env.env.mountain_density_range == (0.18, 0.26)
    assert env.env.num_castles_range == (2, 5)
    assert env.env.castle_val_range == (20, 41)
    assert env.num_opponents == 3
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


def test_factorized_actions_reproduce_flat_teacher_moves():
    context = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    flat = GeneralsPufferEnvironment(
        context=context, board_size=6, horizon=30, opponent="random", teacher="harvester", supervise_teacher=True
    )
    factorized = GeneralsPufferEnvironment(
        context=context,
        board_size=6,
        horizon=30,
        opponent="random",
        teacher="harvester",
        supervise_teacher=True,
        factorized_actions=True,
    )
    flat_observation = flat.reset("73:0:0")
    factorized_observation = factorized.reset("73:0:0")
    assert factorized.spec.action_sizes == [145, 2]
    assert len(factorized_observation.action_masks[0]) == 147
    assert factorized_observation.action_masks[0][-3:] == [True, True, True]
    serializer = NativeEnvironment.__new__(NativeEnvironment)
    serializer.spec = factorized.spec
    transport, action_mask = serializer.encode(factorized_observation)
    assert len(transport) == 4 * factorized.spec.transport_size
    assert len(action_mask) == sum(factorized.spec.action_sizes)

    for _ in range(12):
        flat_target = flat_observation.teachers[0]
        factorized_target = factorized_observation.teachers[0]
        flat_index = int(np.argmax(flat_target.probabilities))
        move_index = int(np.argmax(factorized_target.probabilities[:145]))
        split = int(np.argmax(factorized_target.probabilities[145:]))
        assert factorized_target.weights[0] == 1.0
        assert factorized_target.weights[1] == (0.0 if move_index == 144 else 1.0)
        assert np.array_equal(np.asarray(decode_action(flat_index, 6)), np.asarray(decode_action(move_index, 6, split)))

        flat_transition = flat.step([[flat_index]])
        factorized_transition = factorized.step([[move_index, split]])
        assert flat_transition.rewards == factorized_transition.rewards
        assert flat_transition.score == factorized_transition.score
        assert flat_transition.observation.values == factorized_transition.observation.values
        flat_observation = flat_transition.observation
        factorized_observation = factorized_transition.observation
        if flat_transition.episode_done:
            break

    flat.close()
    factorized.close()

    evaluating = context.model_copy(update={"mode": "evaluate"})
    eval_env = GeneralsPufferEnvironment(
        context=evaluating, board_size=6, teacher="harvester", supervise_teacher=True, factorized_actions=True
    )
    eval_observation = eval_env.reset("73:0:0")
    assert eval_observation.teachers[0].weights == [0.0, 0.0]
    assert sum(eval_observation.teachers[0].probabilities) == 0.0
    serializer.spec = eval_env.spec
    serializer.encode(eval_observation)
    eval_env.close()


def test_batched_games_step_together_and_obey_native_contract():
    context = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    env = BatchedGeneralsPufferEnvironment(
        context=context,
        board_size=6,
        horizon=12,
        opponent="mixed",
        teacher="harvester",
        supervise_teacher=True,
        factorized_actions=True,
        parallel_games=3,
        require_gpu=False,
    )
    observation = env.reset("73:0:0")
    assert env.spec.agents == 3
    assert len(observation.values) == len(observation.action_masks) == len(observation.teachers) == 3
    serializer = NativeEnvironment.__new__(NativeEnvironment)
    serializer.spec = env.spec
    transport, action_mask = serializer.encode(observation)
    assert len(transport) == 4 * env.spec.agents * env.spec.transport_size
    assert len(action_mask) == env.spec.agents * sum(env.spec.action_sizes)

    pass_index = env.spec.action_sizes[0] - 1
    for turn in range(12):
        transition = env.step([[pass_index, 0]] * env.spec.agents)
        assert transition.episode_done is (turn == 11)
        assert len(transition.rewards) == len(transition.terminated) == env.spec.agents
        assert np.isfinite(transition.rewards).all()
        serializer.encode(transition.observation)
    assert transition.terminated == [True] * env.spec.agents
    assert -1 <= transition.score <= 1
    env.close()


def test_training_restarts_a_finished_game_without_ending_the_batch():
    context = EnvironmentContext(seed=73, index=0, mode="train", output=Path("/tmp"))
    env = BatchedGeneralsPufferEnvironment(
        context=context,
        board_size=6,
        horizon=160,
        opponent="harvester",
        teacher="harvester",
        supervise_teacher=True,
        factorized_actions=True,
        parallel_games=1,
        require_gpu=False,
    )
    env.reset("73:0:0")
    for _ in range(160):
        transition = env.step([[144, 0]])
        if transition.terminated[0]:
            break
    else:
        pytest.fail("Scripted opponent did not finish the game before the group horizon")

    assert not transition.episode_done
    assert env.completed[0] == 1
    assert int(env.states.time[0]) == 0
    assert transition.observation.teachers[0].weights[0] == 1.0
    assert not env.step([[144, 0]]).terminated[0]
    assert int(env.states.time[0]) == 1
    env.close()

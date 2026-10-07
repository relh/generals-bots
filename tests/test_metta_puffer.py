"""Classic device transitions, capture rewards and terminal observation recycling."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("metta_training", reason="Pinned private framework required for native environment")
from metta_training.environment import EnvironmentContext
from generals.core import game
from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment


@pytest.mark.parametrize("force_shaping", [False, True])
@pytest.mark.parametrize("objective,expected", [("win_only", [0, .5]), ("signed", [-.5, .5])])
def test_capital_capture_rewards_and_recycled_public_view(tmp_path, objective, expected, force_shaping):
    env = BatchedGeneralsSelfPlayPufferEnvironment(
        context=EnvironmentContext(seed=1, index=0, mode="train", output=tmp_path),
        parallel_games=2, coworld_pool_size=16, require_gpu=False,
        shaping_weight=.25 if force_shaping else 0.0, terminal_reward_mode=objective,
        monotone_force_potential=force_shaping,
    )
    try:
        env.reset_device("capital-capture-proof")
        grid = jnp.zeros((21, 21), jnp.int32).at[1, 1].set(1).at[19, 19].set(2)
        state = game.create_initial_state(grid)
        ownership = state.ownership.at[1, 1, 2].set(True)
        state = state._replace(
            ownership=ownership, ownership_neutral=~ownership.any(axis=0),
            armies=state.armies.at[1, 1].set(20).at[19, 19].set(20).at[1, 2].set(30),
            time=jnp.int32(100),
        )
        env.states = jax.tree.map(lambda x: jnp.stack((x, x)), state)
        left_attack = 2 * 441 + 1 * 21 + 2
        actions = jnp.array([[3528], [left_attack], [3528], [left_attack]], jnp.int32)
        values, masks, rewards, done, truncated = env.step_device(actions)
        if force_shaping:
            from integrations.monotone_force import potential
            info = game.get_info(state)
            army, land = np.asarray(info.army), np.asarray(info.land)
            phi = (.5 * (army-army[::-1])/(army.sum()+1)
                   + .3 * (land-land[::-1])/(land.sum()+1)
                   + .05 * np.asarray(potential(state.armies, state.ownership, state.time, jnp)))
            expected = (np.asarray(expected)-.125*phi).tolist()
        np.testing.assert_allclose(rewards, expected * 2, rtol=1e-6, atol=1e-7)
        assert np.asarray(done).all() and truncated is False
        assert values.shape == (4, 7056) and masks.shape == (4, 3529)
        assert np.isfinite(np.asarray(values)).all() and np.asarray(masks).any(axis=1).all()
        assert not np.asarray(env.states.time).any()
        actual_values, actual_masks = env._observe_both(env.states)
        np.testing.assert_array_equal(values, np.asarray(actual_values).reshape(4, -1))
        np.testing.assert_array_equal(masks, np.asarray(actual_masks).reshape(4, -1))
        np.testing.assert_array_equal(np.sort(np.asarray(env.sides)), [0, 1])
        assert env.base.env.coworld_classic_rules
    finally:
        env.close()

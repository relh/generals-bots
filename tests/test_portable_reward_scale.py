"""Bounded pilot reward contract and actual Classic terminal scaling."""
import numpy as np
import pytest

from integrations.portable_classic_pilot import configure_reward_scale

OPTIONS = dict(terminal_reward_mode='win_only', shaping_weight=.25,
               shaping_gamma=.999, army_shaping_weight=.5, land_shaping_weight=.3,
               castle_shaping_weight=0., imitation_weight=0., land_gain_reward_weight=0.,
               reward_scale=1.)


def test_bound_and_explicit_control():
    for scale, bound in ((1., 1.2), (.5, .6)):
        options = OPTIONS.copy()
        assert configure_reward_scale(options, str(scale)) == pytest.approx(bound)
        assert options == OPTIONS | {'reward_scale': scale}
    for value in ('nan', 'inf', '0', '-1', '0.1'):
        with pytest.raises(ValueError):
            configure_reward_scale(OPTIONS.copy(), value)
    for key in ('land_gain_reward_weight', 'imitation_weight', 'frontier_shaping_weight'):
        with pytest.raises(ValueError):
            configure_reward_scale(OPTIONS | {key: .1}, '.5')


def test_half_reward_preserves_classic_capture_draw_live_and_recycle(tmp_path):
    import jax
    import jax.numpy as jnp
    from metta_training.environment import EnvironmentContext

    from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment

    environments = []
    try:
        for scale in (1., .5):
            env = BatchedGeneralsSelfPlayPufferEnvironment(
                context=EnvironmentContext(seed=1, index=0, mode='train', output=tmp_path),
                parallel_games=4, require_gpu=False, coworld_classic=True,
                coworld_pool_size=16, balance_opponent_sides=True,
                **(OPTIONS | {'reward_scale': scale}))
            env.reset_device('reward-scale-terminal-proof')
            environments.append(env)
        states = environments[0].states
        r, c = np.argwhere(np.asarray(states.generals[0] & states.ownership[0, 0]))[0]
        er, ec = np.argwhere(np.asarray(states.generals[0] & states.ownership[0, 1]))[0]
        dr = 1 if r < 17 else -1
        target = (int(r + dr), int(c))
        states = states._replace(
            generals=states.generals.at[0, er, ec].set(False).at[0, *target].set(True),
            passable=states.passable.at[0, *target].set(True),
            ownership=states.ownership.at[0, 0, *target].set(False).at[0, 1, *target].set(True),
            ownership_neutral=states.ownership_neutral.at[0, *target].set(False),
            armies=states.armies.at[0, r, c].set(20).at[0, *target].set(1).at[0, er, ec].set(1000),
            time=jnp.asarray([1, 1999, 1, 1], dtype=states.time.dtype))
        actions = jnp.full((8, 1), 3528, jnp.int32).at[0, 0].set((1 if dr == 1 else 0) * 441 + r * 21 + c)
        for env in environments:
            env.states = states
        raw, scaled = [env.step_device(actions) for env in environments]
        np.testing.assert_allclose(np.asarray(scaled[2]), .5 * np.asarray(raw[2]), atol=1e-7, rtol=0)
        assert np.asarray(raw[2])[0] > 1  # The winning underdog clipped in native Puffer.
        assert np.max(np.abs(np.asarray(scaled[2]))) <= .6
        np.testing.assert_array_equal(np.asarray(raw[3]), [1, 1, 1, 1, 0, 0, 0, 0])
        for index in (0, 1, 3):
            np.testing.assert_array_equal(np.asarray(raw[index]), np.asarray(scaled[index]))
        for left, right in zip(
            jax.tree.leaves(environments[0].states), jax.tree.leaves(environments[1].states), strict=True,
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    finally:
        for env in environments:
            env.close()

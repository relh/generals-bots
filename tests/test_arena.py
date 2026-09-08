"""Rule and termination tests for the instrumented arena."""
import jax
import jax.numpy as jnp
import numpy as np

from generals.core import game
from generals.evaluation.arena import Rules, make_runner, transition


def toward_adjacent_general(obs, key):
    del key
    index = jnp.argmax((obs.generals & obs.owned_cells).reshape(-1))
    row, col = index // obs.armies.shape[1], index % obs.armies.shape[1]
    return jnp.array([0, row, col, jnp.where(col == 0, 3, 2), 0], dtype=jnp.int32)


def test_mutual_deathtouch_draw_freezes_before_timeout():
    grid = jnp.array([[1, 0, 2], [0, 0, 0]], dtype=jnp.int32)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[1, 0].set(10).at[1, 2].set(10),
                           ownership=state.ownership.at[0, 1, 2].set(True).at[1, 1, 0].set(True),
                           ownership_neutral=state.ownership_neutral.at[1, 0].set(False).at[1, 2].set(False))
    def attack(obs, key):
        index = jnp.argmax((obs.owned_cells & ~obs.generals).reshape(-1))
        return jnp.array([0, index // 3, index % 3, 0, 0], dtype=jnp.int32)
    run = make_runner(attack, attack, Rules(max_turns=20, deathtouch_turn=0), from_states=True)
    states = jax.tree.map(lambda x: jnp.stack([x, x]), state)
    result = run(states, jax.random.split(jax.random.PRNGKey(0), 2), jnp.array([0, 1]))
    np.testing.assert_array_equal(result.finished, [True, True])
    np.testing.assert_array_equal(result.state.winner, [-1, -1])
    np.testing.assert_array_equal(result.state.time, [1, 1])
    np.testing.assert_array_equal(result.counters[:, :, 4], [[0, 0], [0, 0]])


def test_pass_only_games_timeout_without_becoming_terminal():
    idle = lambda obs, key: jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
    grid = jnp.array([[1, 0], [0, 2]], dtype=jnp.int32)
    result = make_runner(idle, idle, Rules(max_turns=7))(
        grid[None], jax.random.PRNGKey(1)[None], jnp.array([0]))
    assert not bool(result.finished[0])
    assert int(result.state.time[0]) == 7
    np.testing.assert_array_equal(result.counters[0, :, 0], [7, 7])


def test_build_rule_spends_army_and_creates_castle():
    grid = jnp.array([[1, 0], [0, 2]], dtype=jnp.int32)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[1, 0].set(60),
                           ownership=state.ownership.at[0, 1, 0].set(True),
                           ownership_neutral=state.ownership_neutral.at[1, 0].set(False))
    actions = jnp.array([[2, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    result, _ = transition(state, actions, Rules(build_castles=True))
    assert bool(result.castles[1, 0])
    assert int(result.armies[1, 0]) == 13  # 35 + 12 adjacent-to-general cost.


def test_malformed_actions_are_rejected_and_counted():
    from generals.evaluation.arena import action_counters
    grid = jnp.array([[1, 0], [0, 2]], dtype=jnp.int32)
    state = game.create_initial_state(grid)
    state = state._replace(armies=state.armies.at[0, 0].set(10))
    for malformed in ([0, 0, 0, 99, 0], [2, 0, 0, 3, 0], [9, 0, 0, 3, 0],
                      [0, 0, 0, 3, 2], [0, 0, .5, 3, 0]):
        actions = jnp.array([malformed, [1, 0, 0, 0, 0]])
        after, _ = transition(state, actions, Rules())
        np.testing.assert_array_equal(after.armies, state.armies)
        np.testing.assert_array_equal(after.ownership, state.ownership)
        counters = action_counters(state, after, actions)
        assert int(counters[0, 5]) == 1

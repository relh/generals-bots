"""Tactical contract checks; strength is measured separately in the arena."""
import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_agent import SentinelAgent, _build_prices, _distance
from generals.core.observation import Observation


def board(owned, enemies=(), castles=(), mountains=(), *, shape=(4, 4), time=0):
    armies = jnp.zeros(shape, jnp.int32)
    mine = jnp.zeros(shape, bool)
    opponent = jnp.zeros(shape, bool)
    generals = jnp.zeros(shape, bool)
    for r, c, army, general in owned:
        armies = armies.at[r, c].set(army)
        mine = mine.at[r, c].set(True)
        generals = generals.at[r, c].set(general)
    for r, c, army, general in enemies:
        armies = armies.at[r, c].set(army)
        opponent = opponent.at[r, c].set(True)
        generals = generals.at[r, c].set(general)
    cities = jnp.zeros(shape, bool)
    for r, c, army in castles:
        cities = cities.at[r, c].set(True)
        armies = armies.at[r, c].set(army)
    walls = jnp.zeros(shape, bool)
    for r, c in mountains:
        walls = walls.at[r, c].set(True)
    return Observation(armies, generals, cities, walls, ~(mine | opponent | walls),
                       mine, opponent, jnp.zeros(shape, bool), jnp.zeros(shape, bool),
                       mine.sum(), (armies * mine).sum(), opponent.sum(),
                       (armies * opponent).sum(), jnp.int32(time))


def action(obs, **kwargs):
    return np.asarray(SentinelAgent(**kwargs).act(obs, jax.random.PRNGKey(0)))


def test_takes_winning_general_capture():
    obs = board([(0, 0, 5, True), (2, 2, 20, False)], [(2, 3, 10, True)])
    np.testing.assert_array_equal(action(obs), [0, 2, 2, 3, 0])


def test_does_not_launch_a_losing_attack():
    obs = board([(0, 0, 4, True)], [(0, 1, 20, True)], mountains=[(1, 0)])
    assert action(obs)[0] == 1


def test_deathtouch_ignores_general_army_only_when_enabled():
    obs = board([(0, 0, 5, True), (2, 2, 2, False)], [(2, 3, 100, True)], time=800)
    np.testing.assert_array_equal(action(obs, deathtouch_turn=800), [0, 2, 2, 3, 0])
    assert not np.array_equal(action(obs), [0, 2, 2, 3, 0])


def test_intercepts_threat_to_general_from_third_tile():
    obs = board([(1, 1, 3, True), (0, 2, 15, False)], [(1, 2, 10, False)])
    np.testing.assert_array_equal(action(obs), [0, 0, 2, 1, 0])


def test_deathtouch_defense_requires_interception():
    obs = board([(1, 1, 100, True), (0, 2, 15, False)], [(1, 2, 10, False)], time=800)
    np.testing.assert_array_equal(action(obs, deathtouch_turn=800), [0, 0, 2, 1, 0])


def test_reinforces_outmatched_general():
    obs = board([(1, 1, 3, True), (1, 0, 15, False)], [(1, 2, 10, False)])
    np.testing.assert_array_equal(action(obs), [0, 1, 0, 3, 0])


def test_affordable_castle_capture():
    obs = board([(0, 0, 10, True), (2, 2, 30, False)], castles=[(2, 3, 10)])
    np.testing.assert_array_equal(action(obs), [0, 2, 2, 3, 0])


def test_builds_safe_profitable_castle_and_respects_rules():
    obs = board([(0, 0, 20, True), (2, 2, 80, False)])
    assert action(obs, build_castles=True)[0] == 2
    assert action(obs)[0] != 2
    late = obs._replace(timestep=jnp.int32(1150))
    assert action(late, build_castles=True)[0] != 2


def test_build_cost_matches_modifier():
    from generals.core.game import create_initial_state
    from generals.modifiers.build_castles import build_cost_grid
    # create_initial_state's grid uses 1/2 for player generals and >2 for castles.
    state = create_initial_state(jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], jnp.int32))
    structures = state.ownership[0] & (state.generals | state.castles)
    np.testing.assert_array_equal(_build_prices(structures), build_cost_grid(state, 0))


def test_routes_around_walls_and_vmaps():
    obs = board([(0, 0, 20, True), (2, 1, 20, False)], [(2, 3, 5, True)], mountains=[(2, 2)])
    batch = jax.tree.map(lambda x: jnp.stack([x, x]) if x is not None else None, obs)
    agent = SentinelAgent()
    actions = jax.jit(jax.vmap(agent.act))(batch, jax.random.split(jax.random.PRNGKey(0), 2))
    assert actions.shape == (2, 5)
    assert not np.array_equal(actions[0], [0, 2, 1, 3, 0])
    passable = ~obs.mountains
    d = _distance(passable, obs.opponent_cells)
    assert d[2, 1] == 4


def test_breaks_adjacent_general_standoff_with_safe_founding_move():
    obs = board([(1, 1, 2, True)], [(2, 1, 2, True)])
    picked = action(obs)
    assert picked[0] == 0
    assert picked[3] != 1  # Found land rather than attack the equal general.
    stronger = board([(1, 1, 2, True)], [(2, 1, 3, True)])
    assert action(stronger)[0] == 1  # Cannot afford expansion under a full attack.


def test_dispatches_surplus_through_two_step_enemy_corridor():
    obs = board([(3, 1, 8, True)], [(2, 2, 8, True)], mountains=[(3, 0), (3, 2)])
    np.testing.assert_array_equal(action(obs), [0, 3, 1, 0, 1])

"""Actual engine growth and deployment accounting for branch collection."""
import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_v20_agent import SentinelV20Agent
from generals.core import game
from tests.test_multiplayer import board as state_board, give
from tests.test_sentinel_v20_agent import AGENT, PROPOSE, started, owned_move


def test_actual_engine_growth_does_not_refresh_or_double_credit_collection():
    state = state_board({0: (0, 0), 1: (0, 6)}, size=7)
    for cell, army in [((0, 0), 100), ((4, 4), 5), ((3, 4), 4), ((5, 4), 4), ((4, 5), 1)]:
        state = give(state, 0, cell, army)
    state = give(state, 1, (4, 6), 7)._replace(time=jnp.int32(49))
    agent = SentinelV20Agent(deathtouch_turn=800)
    memory = agent.initial_memory((7, 7))
    key = jax.random.PRNGKey(17)
    action, memory, tel = agent.step(game.get_observation(state, 0), key, memory)
    assert tel['branch_started']
    np.testing.assert_array_equal(action, [0, 3, 4, 1, 0])
    assert memory.expected_army == 8 and memory.budget == 1
    expiry = memory.expires
    state, _ = game.step(state, jnp.stack((action, jnp.array([1, 0, 0, 0, 0], jnp.int32))))
    assert state.time == 50 and state.armies[4, 4] == 9 and state.armies[3, 4] == 2
    action, returned, tel = agent.step(game.get_observation(state, 0), key, memory)
    assert tel['branch_continued'] and returned.expires == expiry
    assert returned.budget == 0 and returned.expected_army == 13
    np.testing.assert_array_equal(action, [0, 5, 4, 0, 0])
    state, _ = game.step(state, jnp.stack((action, jnp.array([1, 0, 0, 0, 0], jnp.int32))))
    assert state.armies[4, 4] == 13 and state.armies[5, 4] == 1


def test_deployment_cannot_borrow_army_remaining_at_old_rally():
    obs, memory = started()
    action, memory, _ = PROPOSE(obs, memory)
    obs = owned_move(obs, action)
    action, memory, _ = PROPOSE(obs, memory)
    obs = owned_move(obs, action)
    assert memory.packet != memory.rally
    obs = obs._replace(armies=obs.armies.at[4, 4].set(100).at[4, 5].set(2))
    memory = memory._replace(expected_army=jnp.int32(2))
    action, returned, tel = PROPOSE(obs, memory)
    np.testing.assert_array_equal(action, [1, 0, 0, 0, 0])
    assert not tel['branch_continued'] and returned.phase == 0

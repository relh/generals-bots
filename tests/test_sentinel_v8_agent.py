"""Action-cost changes must deliver real packets and preserve frozen ablations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.sentinel_v7_agent import SentinelV7Agent
from generals.agents.sentinel_v8_agent import SentinelV8Agent, _select_collection
from generals.core import game
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import KEY, intercept_board, legal
from tests.test_sentinel_v6_agent import initial_state
from tests.test_sentinel_v7_agent import concentration_board

PASS = jnp.array([1, 0, 0, 0, 0])


def same_tree(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)


def test_cheapest_source_is_lexicographic_and_only_among_sufficient_routes():
    allowed = jnp.array([[False, True, True, True]])
    distance = jnp.array([[1, 2, 2, 6]])
    delivery = jnp.array([[2, 12, 13, 1000]])
    assert _select_collection(allowed, distance, delivery, 2, True) == 2
    assert _select_collection(allowed, distance, delivery, 2, False) == 3
    # Equal-cost ties retain original v7 delivery preference, then flat index.
    assert _select_collection(allowed, distance, delivery.at[0, 1].set(13), 2, True) == 1


@pytest.mark.parametrize("cheap,direct", [(True, False), (False, True), (True, True)])
def test_cost_modes_execute_shorter_real_owned_route_and_capture(cheap, direct):
    state = initial_state(concentration_board())
    agent = SentinelV8Agent(cheapest_collection=cheap, direct_deployment=direct)
    memory = agent.initial_memory(state.armies.shape)
    expected = ([[0, 5, 4, 0, 0]] if not direct else []) + [[0, 4, 4, 3, 0], [0, 4, 5, 0, 0]]
    expiry = None
    for i, move in enumerate(expected):
        obs = game.get_observation(state, 0)
        action, memory, tel = agent.step(obs, KEY, memory)
        np.testing.assert_array_equal(action, move)
        legal(obs, action)
        if i == 0:
            assert bool(tel["offense_direct_started"]) == direct
            if direct:
                assert tel["offense_direct_available"] and tel["offense_direct_chosen"]
                assert tel["offense_direct_eta"] == 2
                assert tel["offense_deploying"] and not tel["offense_collecting"]
                assert memory.phase == 2 and memory.remaining == 1
                assert memory.expires == obs.timestep + 3
            else:
                assert tel["offense_collecting"] and tel["offense_remaining"] == 1
                assert tel["offense_delivered"] == 9
            expiry = int(memory.expires)
        if i < len(expected) - 1:
            assert memory.expires == expiry
        state, _ = game.step(state, jnp.stack((action, PASS)))
    assert tel["offense_attack_issued"] and memory.phase == 0
    assert state.ownership[0, 3, 5] and state.armies[3, 5] == (4 if direct else 7)
    # The ready packet does not borrow any unissued side-donor transfers.
    if direct:
        assert state.armies[6, 2] == 6 and state.armies[5, 4] == 4


def ready_board(army=6, target=2, general=False, time=200):
    return board(
        [(8, 8, 20, True), (4, 4, army, False), (4, 5, 1, False)],
        [(3, 5, target, general), (0, 0, 10, not general)],
        shape=(9, 9),
        time=time,
    )


def test_direct_does_not_require_a_useful_collection_source_or_rally_gain():
    obs = ready_board()
    agent = SentinelV8Agent()
    action, memory, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(action, [0, 4, 4, 3, 0])
    assert tel["offense_collection_eta"] == -1
    assert tel["offense_direct_started"] and tel["offense_deploying"]
    assert memory.phase == 2 and memory.expected_army == 6
    legal(obs, action)


def test_ready_force_must_pay_each_departure_and_current_counterforce():
    # Other owned surplus keeps the unchanged target selector eligible, but
    # no single packet meets target2 + two departures + safety1 = five.
    obs = board(
        [(8, 8, 20, True), (4, 4, 4, False), (4, 5, 1, False), (5, 4, 2, False), (5, 3, 2, False)],
        [(3, 5, 2, False), (0, 0, 10, True)],
        shape=(9, 9),
        time=200,
    )
    agent = SentinelV8Agent()
    _, _, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    assert not tel["offense_direct_available"]
    obs = ready_board()._replace(
        opponent_cells=ready_board().opponent_cells.at[3, 6].set(True),
        neutral_cells=ready_board().neutral_cells.at[3, 6].set(False),
        armies=ready_board().armies.at[3, 6].set(8),
        castles=ready_board().castles.at[3, 5].set(True),
    )
    _, _, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    assert not tel["offense_direct_available"]


@pytest.mark.parametrize("concentrate", [True, False])
def test_disabled_exact_reference_actions_all_memory_and_telemetry(concentrate):
    v7 = SentinelV7Agent(concentrate_armies=concentrate)
    v8 = SentinelV8Agent(concentrate_armies=concentrate, cheapest_collection=False, direct_deployment=False)
    state = initial_state(concentration_board() if concentrate else intercept_board())
    m7, m8 = v7.initial_memory(state.armies.shape), v8.initial_memory(state.armies.shape)
    for _ in range(3):
        obs = game.get_observation(state, 0)
        a7, m7, t7 = v7.step(obs, KEY, m7)
        a8, m8, t8 = v8.step(obs, KEY, m8)
        same_tree((a7, m7, t7), (a8, m8, t8))
        state, _ = game.step(state, jnp.stack((a7, PASS)))


@pytest.mark.parametrize("change", ["lost_packet", "lost_route", "depleted", "gap", "expired"])
def test_direct_revalidates_actual_packet_route_force_and_fixed_age(change):
    agent = SentinelV8Agent()
    state = initial_state(ready_board())
    action, memory, _ = agent.step(game.get_observation(state, 0), KEY, agent.initial_memory(state.armies.shape))
    state, _ = game.step(state, jnp.stack((action, PASS)))
    obs = game.get_observation(state, 0)
    if change == "lost_packet":
        obs = obs._replace(owned_cells=obs.owned_cells.at[4, 5].set(False))
    elif change == "lost_route":
        obs = obs._replace(opponent_cells=obs.opponent_cells.at[3, 5].set(False))
    elif change == "depleted":
        obs = obs._replace(armies=obs.armies.at[4, 5].set(2))
    elif change == "gap":
        obs = obs._replace(timestep=obs.timestep + 1)
    else:
        memory = memory._replace(expires=obs.timestep - 1)
    action, new, tel = agent.step(obs, KEY, memory)
    assert tel["offense_released"] and not tel["offense_continued"] and new.phase == 0
    legal(obs, action)


def test_defense_preempts_ready_offense_and_nested_memory_matches_issued_action():
    obs = intercept_board()
    agent = SentinelV8Agent()
    memory = agent.initial_memory(obs.armies.shape)._replace(
        packet=jnp.int32(14),
        rally=jnp.int32(15),
        objective=jnp.int32(9),
        phase=jnp.int32(2),
        remaining=jnp.int32(2),
        expires=jnp.int32(240),
        last_turn=jnp.int32(231),
        expected_army=jnp.int32(14),
    )
    expected, defense, _ = agent.base.step(obs, KEY, memory.defense)
    action, new, tel = agent.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, expected)
    same_tree(defense, new.defense)
    assert tel["offense_defense_priority"] and not tel["offense_direct_started"] and new.phase == 0


def test_winning_capture_and_projected_deathtouch_remain_priorities():
    obs = board([(5, 3, 3, True), (2, 2, 2, False)], [(1, 3, 50, False), (2, 3, 100, True)], shape=(6, 6), time=800)
    agent = SentinelV8Agent(deathtouch_turn=800)
    action, _, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    assert tel["offense_defense_priority"]
    legal(obs, action)


def test_jit_vmap_fixed_memory_and_independent_direct_plan():
    agent = SentinelV8Agent()
    a, b = ready_board(), ready_board(army=1)
    observations = jax.tree.map(lambda x, y: None if x is None else jnp.stack((x, y)), a, b)
    m = agent.initial_memory(a.armies.shape)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), m)
    actions, new, tel = jax.jit(jax.vmap(agent.step))(observations, jax.random.split(KEY, 2), memories)
    assert actions.shape == (2, 5) and tel["offense_direct_started"][0]
    assert new.phase[0] == 2 and new.phase[1] == 0
    assert len(jax.tree.leaves(new)) == 15 and all(x.dtype == jnp.int32 for x in jax.tree.leaves(new))


def test_available_faster_packet_failing_home_safety_is_not_issued():
    open_cells = {(r, 3) for r in range(1, 7)} | {(4, c) for c in range(2, 7)}
    walls = [(r, c) for r in range(7) for c in range(7) if (r, c) not in open_cells]
    obs = board(
        [
            (6, 3, 5, True),
            (4, 3, 20, False),
            (4, 2, 20, False),  # Safe campaign reinforcement; farther from offense target.
            (5, 3, 1, False),
            (3, 3, 1, False),
            (2, 3, 1, False),
            (4, 4, 1, False),
            (4, 5, 1, False),
        ],
        [(1, 3, 20, False), (4, 6, 2, False)],
        castles=[(4, 6, 2)],
        mountains=walls,
        shape=(7, 7),
        time=201,
    )
    agent = SentinelV8Agent()
    memory = agent.initial_memory(obs.armies.shape)
    expected, defense, before = agent.base.step(obs, KEY, memory.defense)
    action, new, tel = agent.step(obs, KEY, memory)
    assert tel["offense_direct_available"] and tel["offense_direct_chosen"]
    assert not tel["offense_home_coverage_safe"] and not tel["offense_direct_started"]
    assert not before["intercept_guard"] and not tel["offense_defense_priority"]
    np.testing.assert_array_equal(expected, [0, 4, 2, 3, 0])
    np.testing.assert_array_equal(action, expected)
    same_tree(new.defense, defense)
    assert new.phase == 0
    legal(obs, action)


def test_direct_affordability_projects_deathtouch_to_arrival():
    obs = board(
        [(8, 8, 1000, True), (4, 4, 3, False), (4, 5, 1, False), (5, 4, 80, False), (6, 4, 80, False)],
        [(3, 5, 100, True)],
        shape=(9, 9),
        time=799,
    )
    agent = SentinelV8Agent(deathtouch_turn=800)
    action, memory, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(action, [0, 4, 4, 3, 0])
    assert tel["offense_direct_started"] and tel["offense_direct_eta"] == 2
    assert tel["offense_required"] == 3 and memory.phase == 2
    legal(obs, action)

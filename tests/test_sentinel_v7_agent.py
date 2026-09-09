"""Offensive concentration must deliver and attack, not just enlarge a rally."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.sentinel_agent import _distance
from generals.agents.sentinel_v6_agent import SentinelV6Agent
from generals.agents.sentinel_v7_agent import SentinelV7Agent, _collection_route, _preserves_home
from generals.core import game
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import KEY, intercept_board, legal
from tests.test_sentinel_v6_agent import initial_state


def concentration_board():
    # Same owned donor amounts and four-step geometry as recorded game8 t200;
    # a compact tactical board isolates collection and subsequent deployment.
    return board(
        [
            (8, 8, 20, True),
            (6, 2, 6, False),
            (6, 3, 2, False),
            (5, 3, 4, False),
            (5, 4, 4, False),
            (4, 4, 6, False),
            (4, 5, 2, False),
        ],
        [(3, 5, 2, False), (0, 0, 10, True)],
        shape=(9, 9),
        time=200,
    )


def test_collects_real_donors_then_deploys_and_captures():
    state = initial_state(concentration_board())
    agent = SentinelV7Agent()
    memory = agent.initial_memory(state.armies.shape)
    expected = [[0, 6, 2, 3, 0], [0, 6, 3, 0, 0], [0, 5, 3, 3, 0], [0, 5, 4, 0, 0], [0, 4, 4, 3, 0], [0, 4, 5, 0, 0]]
    expiry = None
    for turn, move in enumerate(expected):
        obs = game.get_observation(state, 0)
        action, memory, tel = agent.step(obs, KEY, memory)
        np.testing.assert_array_equal(action, move)
        legal(obs, action)
        assert bool(tel["offense_collecting"]) == (turn < 4)
        assert bool(tel["offense_deploying"]) == (turn >= 4)
        if turn == 0:
            assert tel["offense_delivered"] == 18
            expiry = int(memory.expires)
        if turn < 5:
            assert memory.expires == expiry
            assert memory.defense.defender == -1
        state, _ = game.step(state, jnp.stack((action, jnp.array([1, 0, 0, 0, 0]))))
        if turn == 3:
            assert state.armies[4, 4] == 18
    assert tel["offense_attack_issued"] and memory.phase == 0
    assert state.ownership[0, 3, 5] and state.armies[3, 5] == 16


def test_collection_arithmetic_counts_each_garrison_once_and_cannot_cross_a_gap():
    obs = concentration_board()
    eligible = obs.owned_cells & ~obs.generals
    target = jnp.zeros_like(eligible).at[4, 4].set(True)
    distance, delivered, largest, _ = _collection_route(obs.armies, eligible, target)
    assert distance[6, 2] == 4 and delivered[6, 2] == 18 and largest[6, 2] == 6
    broken = eligible.at[5, 3].set(False)
    distance, delivered, _, _ = _collection_route(obs.armies, broken, target)
    assert distance[6, 2] >= 1e6 and delivered[6, 2] < 0


@pytest.mark.parametrize(
    "change", ["lost_packet", "depleted", "fog_objective", "lost_route", "expired", "gap", "grown_target"]
)
def test_plan_revalidates_ownership_force_route_and_age(change):
    state = initial_state(concentration_board())
    agent = SentinelV7Agent()
    action, memory, _ = agent.step(game.get_observation(state, 0), KEY, agent.initial_memory(state.armies.shape))
    state, _ = game.step(state, jnp.stack((action, jnp.array([1, 0, 0, 0, 0]))))
    obs = game.get_observation(state, 0)
    if change == "lost_packet":
        obs = obs._replace(owned_cells=obs.owned_cells.at[6, 3].set(False))
    elif change == "depleted":
        obs = obs._replace(armies=obs.armies.at[6, 3].set(2))
    elif change == "fog_objective":
        obs = obs._replace(
            opponent_cells=obs.opponent_cells.at[3, 5].set(False), fog_cells=obs.fog_cells.at[3, 5].set(True)
        )
    elif change == "lost_route":
        obs = obs._replace(owned_cells=obs.owned_cells.at[5, 3].set(False))
    elif change == "grown_target":
        obs = obs._replace(armies=obs.armies.at[3, 5].set(100))
    elif change == "expired":
        memory = memory._replace(expires=obs.timestep - 1)
    elif change == "gap":
        obs = obs._replace(timestep=obs.timestep + 1)
    action, new, tel = agent.step(obs, KEY, memory)
    assert tel["offense_released"] and not tel["offense_continued"]
    assert new.phase == 0
    legal(obs, action)


def test_disabled_preserves_v6_defender_progress_and_original_telemetry():
    v6, v7 = SentinelV6Agent(), SentinelV7Agent(concentrate_armies=False)
    a = intercept_board()
    m6, m7 = v6.initial_memory(a.armies.shape), v7.initial_memory(a.armies.shape)
    for obs in [a, a._replace(armies=a.armies.at[2, 2].set(1).at[2, 3].set(14), timestep=a.timestep + 1)]:
        expected, m6, before = v6.step(obs, KEY, m6)
        action, m7, tel = v7.step(obs, KEY, m7)
        np.testing.assert_array_equal(action, expected)
        for x, y in zip(m6, m7.defense):
            np.testing.assert_array_equal(x, y)
        assert tel.keys() == before.keys()
        for k in tel:
            np.testing.assert_array_equal(tel[k], before[k])
        assert m7.phase == 0


def test_transfer_cannot_drain_a_screen_even_without_a_changed_guard_action():
    open_cells = {(r, 3) for r in range(1, 7)} | {(4, 4), (4, 5)}
    walls = [(r, c) for r in range(7) for c in range(7) if (r, c) not in open_cells]
    obs = board(
        [
            (6, 3, 5, True),
            (4, 3, 20, False),
            (5, 3, 1, False),
            (3, 3, 1, False),
            (2, 3, 1, False),
            (4, 4, 1, False),
            (4, 5, 10, False),
        ],
        [(1, 3, 20, False)],
        mountains=walls,
        shape=(7, 7),
        time=201,
    )
    terrain = ~(obs.mountains | obs.structures_in_fog)
    home = obs.owned_cells & obs.generals
    dist = _distance(terrain, home)
    assert not _preserves_home(obs, jnp.array([0, 4, 3, 3, 0]), terrain, home, dist, None)
    agent = SentinelV7Agent()
    memory = agent.initial_memory(obs.armies.shape)._replace(
        packet=jnp.int32(31),
        rally=jnp.int32(33),
        objective=jnp.int32(10),
        phase=jnp.int32(1),
        remaining=jnp.int32(2),
        expires=jnp.int32(210),
        last_turn=jnp.int32(200),
        expected_army=jnp.int32(20),
    )
    expected, _, before = agent.base.step(obs, KEY, memory.defense)
    assert not before["intercept_guard"]
    action, new, tel = agent.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, expected)
    assert not tel["offense_home_coverage_safe"] and new.phase == 0
    assert _preserves_home(obs, jnp.array([0, 4, 5, 2, 0]), terrain, home, dist, None)


def test_defense_preempts_offense_without_claiming_an_unissued_defender_move():
    obs = intercept_board()
    agent = SentinelV7Agent()
    memory = agent.initial_memory(obs.armies.shape)._replace(
        packet=jnp.int32(14),
        rally=jnp.int32(15),
        objective=jnp.int32(9),
        phase=jnp.int32(1),
        remaining=jnp.int32(1),
        expires=jnp.int32(240),
        last_turn=jnp.int32(231),
        expected_army=jnp.int32(14),
    )
    expected, defender, _ = agent.base.step(obs, KEY, memory.defense)
    action, new, tel = agent.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, expected)
    assert tel["offense_defense_priority"] and new.phase == 0
    for x, y in zip(new.defense, defender):
        np.testing.assert_array_equal(x, y)


def test_jit_vmap_has_fixed_separate_memory_and_no_benefit_collection_is_skipped():
    agent = SentinelV7Agent()
    a = concentration_board()
    b = a._replace(armies=jnp.where(a.owned_cells & ~a.generals, 1, a.armies))
    observations = jax.tree.map(lambda x, y: jnp.stack((x, y)) if x is not None else None, a, b)
    memory = agent.initial_memory(a.armies.shape)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    actions, new, tel = jax.jit(jax.vmap(agent.step))(observations, jax.random.split(KEY, 2), memories)
    assert actions.shape == (2, 5)
    assert new.phase[0] > 0 and new.phase[1] == 0
    assert all(x.dtype == jnp.int32 for x in jax.tree.leaves(new))


def test_immediate_winning_capture_preempts_an_active_offensive_plan():
    obs = board([(5, 3, 3, True), (2, 2, 2, False)], [(1, 3, 50, False), (2, 3, 100, True)], shape=(6, 6), time=800)
    agent = SentinelV7Agent(deathtouch_turn=800)
    memory = agent.initial_memory(obs.armies.shape)._replace(
        packet=jnp.int32(14),
        rally=jnp.int32(14),
        objective=jnp.int32(9),
        phase=jnp.int32(2),
        remaining=jnp.int32(3),
        expires=jnp.int32(805),
        last_turn=jnp.int32(799),
        expected_army=jnp.int32(2),
    )
    action, new, tel = agent.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    assert tel["offense_defense_priority"] and new.phase == 0
    legal(obs, action)

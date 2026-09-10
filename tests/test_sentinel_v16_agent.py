"""Public same-front allocation, serial force accounting and outer priorities."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_agent import _distance
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v16_agent import SentinelV16Agent, _route_residual
from generals.core import game
from tests.test_multiplayer import board as state_board
from tests.test_multiplayer import give
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import KEY, intercept_board, legal
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
PARENT = SentinelV10Agent(**RULES)
AGENT = SentinelV16Agent(**RULES)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)


def fixture(name):
    data = json.loads((Path(__file__).parent / f"fixtures/campaign_feeder/{name}.json").read_text())
    obs = read_observation(io.StringIO(data["public_wire"]), *data["shape"])
    memory = restore(PARENT.initial_memory(obs.armies.shape), data["incoming_memory"])
    return data, obs, memory, jnp.array(data["action_key"], jnp.uint32)


def telemetry_action(tel, label):
    return jnp.array([tel[f"feeder_{label}_{x}"] for x in ("kind", "row", "column", "direction", "split")])


def test_real_juraj100_commits_actual_reference_branch_with_better_force_per_action():
    data, obs, memory, key = fixture("juraj100")
    action, returned, tel = AGENT.step(obs, key, memory)
    np.testing.assert_array_equal(action, data["expected_reference_action"])
    np.testing.assert_array_equal(telemetry_action(tel, "collector"), data["expected_parent_action"])
    assert tel["feeder_available"] and tel["feeder_started"] and tel["feeder_home_coverage_safe"]
    assert tel["actual_v8_action_issued"] and tel["mobilization_parent_action_issued"]
    assert tel["feeder_eta"] == 9 and tel["feeder_residual"] == 13
    assert tel["feeder_collector_eta"] == 5 and tel["feeder_collector_residual_upper"] >= 4
    assert tel["feeder_residual"] * 5 > tel["feeder_collector_residual_upper"] * 9
    assert returned.base.phase == 3 and returned.base.packet == 10 * 19 + 17
    assert returned.base.remaining == 8 and returned.base.expires == 110
    assert not tel["offense_collecting"] and tel["offense_deploying"]
    legal(obs, action)


@pytest.mark.parametrize("name", ["amin129", "amin142"])
def test_productive_amin_collection_is_not_replaced_by_unrelated_front_packet(name):
    data, obs, memory, key = fixture(name)
    expected, parent_memory, parent_tel = PARENT.step(obs, key, memory)
    action, returned, tel = AGENT.step(obs, key, memory)
    np.testing.assert_array_equal(action, data["expected_parent_action"])
    same_tree((action, returned), (expected, parent_memory))
    same_tree({k: tel[k] for k in parent_tel}, parent_tel)
    assert not tel["feeder_available"] and not tel["feeder_started"]
    assert tel["offense_started"] and tel["offense_collecting"]


@pytest.mark.parametrize("name", ["juraj100", "amin129", "amin142"])
def test_disabled_is_exact_independent_v10_tuple(name):
    _, obs, memory, key = fixture(name)
    disabled = SentinelV16Agent(**RULES, feed_front=False)
    same_tree(disabled.step(obs, key, memory), PARENT.step(obs, key, memory))


def route_state():
    state = state_board({0: (8, 8), 1: (0, 0)}, size=9)
    for cell, army in [((8, 8), 30), ((4, 1), 4), ((4, 2), 3), ((4, 3), 5)]:
        state = give(state, 0, cell, army)
    state = give(state, 1, (4, 4), 2)
    return state._replace(time=jnp.int32(20))


def active_memory(agent, obs, packet=(4, 1), target=(4, 4), eta=3):
    memory = agent.initial_memory(obs.armies.shape)
    width = obs.armies.shape[1]
    base = memory.base._replace(
        packet=jnp.int32(packet[0] * width + packet[1]),
        rally=jnp.int32(packet[0] * width + packet[1]),
        objective=jnp.int32(target[0] * width + target[1]),
        phase=jnp.int32(3), remaining=jnp.int32(eta), expires=obs.timestep + eta + 1,
        last_turn=obs.timestep - 1, expected_army=obs.armies[packet],
    )
    return memory._replace(base=base, last_turn=obs.timestep - 1)


def test_owned_route_collects_once_and_actual_engine_capture_matches_projection():
    state = route_state()
    obs = game.get_observation(state, 0)
    memory = active_memory(AGENT, obs)
    expiry = int(memory.base.expires)
    visited = set()
    for column, expected_army in [(1, 6), (2, 10), (3, 7)]:
        obs = game.get_observation(state, 0)
        action, memory, tel = AGENT.step(obs, KEY, memory)
        np.testing.assert_array_equal(action, [0, 4, column, 3, 0])
        assert tel["feeder_continued"] and tel["feeder_residual"] == 7
        assert not any(bool(value) for name, value in tel.items() if name.startswith("feeder_abort_"))
        assert (4, column) not in visited
        visited.add((4, column))
        legal(obs, action)
        state, _ = game.step(state, jnp.stack((action, PASS)))
        assert state.armies[4, column] == 1
        assert state.armies[4, column + 1] == expected_army
        if column < 3:
            assert memory.base.phase == 3 and memory.base.expires == expiry
            assert memory.base.remaining == 3 - column
    assert state.ownership[0, 4, 4] and memory.base.phase == 0
    assert tel["offense_attack_issued"]


def test_dp_direction_uses_credited_branch_and_does_not_credit_zero_or_off_route_donors():
    obs = board([(5, 5, 20, True), (2, 1, 4, False), (1, 1, 2, False),
                 (1, 2, 2, False), (2, 2, 9, False), (3, 1, 100, False)],
                [(1, 3, 2, False)], shape=(6, 6))
    target = obs.opponent_cells
    distance = _distance((obs.owned_cells | target) & ~obs.generals, target)
    residual, direction = _route_residual(obs.armies, obs.owned_cells, obs.generals, target, distance)
    # Right uses the nine-army donor, whereas UP is equally short but weaker.
    assert direction[2, 1] == 3 and residual[2, 1] == 10
    # The 100 below the source is not on a shortest source->objective route.
    depleted = obs.armies.at[2, 2].set(0)
    remaining, direction = _route_residual(depleted, obs.owned_cells, obs.generals, target, distance)
    assert direction[2, 1] == 0 and remaining[2, 1] == 3


@pytest.mark.parametrize("change", ["lost_packet", "lost_route", "depleted", "expired", "gap", "reset"])
def test_carried_plan_aborts_on_observed_invalidity_without_refreshing_expiry(change):
    obs = game.get_observation(route_state(), 0)
    memory = active_memory(AGENT, obs)
    if change == "lost_packet":
        obs = obs._replace(owned_cells=obs.owned_cells.at[4, 1].set(False))
    elif change == "lost_route":
        obs = obs._replace(owned_cells=obs.owned_cells.at[4, 2].set(False))
    elif change == "depleted":
        obs = obs._replace(armies=obs.armies.at[4, 1].set(2))
    elif change == "expired":
        memory = memory._replace(base=memory.base._replace(expires=obs.timestep - 1))
    elif change == "gap":
        memory = memory._replace(last_turn=obs.timestep - 2)
    else:
        memory = memory._replace(last_turn=obs.timestep)
    action, returned, tel = AGENT.step(obs, KEY, memory)
    assert not tel["feeder_issued"] and returned.base.phase != 3
    if change not in ("gap", "reset"):
        assert tel["feeder_released"]
    legal(obs, action)


def test_active_feeder_preserves_actual_build_and_winning_general_capture():
    obs = board([(8, 8, 30, True), (4, 1, 80, False), (4, 2, 3, False), (4, 3, 5, False)],
                [(4, 4, 2, False)], shape=(9, 9), time=200)
    action, returned, tel = AGENT.step(obs, KEY, active_memory(AGENT, obs))
    assert action[0] == 2 and tel["feeder_abort_priority"] and returned.base.phase == 0
    win = obs._replace(generals=obs.generals.at[4, 4].set(True))
    action, returned, tel = AGENT.step(win, KEY, active_memory(AGENT, win))
    np.testing.assert_array_equal(action, [0, 4, 3, 3, 0])
    assert tel["feeder_abort_priority"] and returned.base.phase == 0


def test_fresh_defense_keeps_its_actual_transport_and_clears_offensive_reservation():
    obs = intercept_board()
    memory = active_memory(AGENT, obs, packet=(2, 2), target=(1, 3), eta=2)
    action, returned, tel = AGENT.step(obs, KEY, memory)
    reference, defender, _ = AGENT.base.base.base.base.step(obs, KEY, memory.base.defense)
    np.testing.assert_array_equal(action, reference)
    same_tree(returned.base.defense, defender)
    assert not tel["feeder_issued"] and returned.base.phase == 0


def test_outer_remembered_pursuit_clears_unissued_phase3_and_masks_inner_events():
    obs = game.get_observation(route_state(), 0)
    memory = active_memory(AGENT, obs)._replace(enemy_general=jnp.int32(1))
    obs = obs._replace(fog_cells=obs.fog_cells.at[0, 1].set(True))
    action, returned, tel = AGENT.step(obs, KEY, memory)
    assert tel["feeder_issued"] and tel["pursuit_override"] and not tel["actual_v8_action_issued"]
    assert returned.base.phase == 0 and returned.enemy_general == 1
    assert not np.array_equal(action, telemetry_action(tel, "proposal"))


def test_jit_vmap_keeps_native19_and_isolates_active_from_lost_packet():
    obs = game.get_observation(route_state(), 0)
    invalid = obs._replace(owned_cells=obs.owned_cells.at[4, 1].set(False))
    batch = jax.tree.map(lambda x, y: None if x is None else jnp.stack((x, y)), obs, invalid)
    memory = active_memory(AGENT, obs)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    _, returned, tel = jax.jit(jax.vmap(AGENT.step))(batch, jnp.stack((KEY, KEY)), memories)
    np.testing.assert_array_equal(tel["feeder_issued"], [True, False])
    leaves = jax.tree.leaves(returned)
    assert len(leaves) == 19 and all(x.dtype == jnp.int32 and x.shape == (2,) for x in leaves)


def test_visible_general_preserves_original_campaign_before_new_feeder_selection():
    _, obs, memory, key = fixture("juraj100")
    # A newly revealed distant general is not an affordable local collection
    # objective. Preserve the original campaign's actual response to it.
    obs = obs._replace(
        opponent_cells=obs.opponent_cells.at[0, 1].set(True),
        generals=obs.generals.at[0, 1].set(True),
        fog_cells=obs.fog_cells.at[0, 1].set(False),
        armies=obs.armies.at[0, 1].set(1000),
    )
    action, returned, tel = AGENT.step(obs, key, memory)
    expected, parent_memory, _ = PARENT.step(obs, key, memory)
    assert not tel["feeder_available"] and not tel["feeder_issued"]
    same_tree((action, returned), (expected, parent_memory))

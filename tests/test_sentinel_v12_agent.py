"""Public-route action checks, separate from complete-game strategy gates."""
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_agent import _build_prices
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v12_agent import SentinelV12Agent, _project_action, _winning
from generals.core import game
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)


def fixture(name):
    d = json.loads((Path(__file__).parent / 'fixtures/route-guard' / (name + '.json')).read_text())
    obs = read_observation(io.StringIO(d['public_wire']), *d['shape'])
    parent = SentinelV10Agent(**RULES)
    memory = restore(parent.initial_memory(obs.armies.shape), d['incoming_memory'])
    return d, obs, jnp.array(d['action_key'], jnp.uint32), memory


def test_recorded_transfer_cannot_open_the_known_cheaper_home_route():
    d, obs, key, memory = fixture('weakening-transfer')
    parent = SentinelV10Agent(**RULES)
    expected, _, _ = parent.step(obs, key, memory)
    np.testing.assert_array_equal(expected, d['expected_parent_action'])
    action, returned, tel = SentinelV12Agent(**RULES).step(obs, key, memory)
    assert tel['route_guard_parent_rejected'] and tel['route_guard_issued']
    assert tel['route_guard_deficit_before'] == 14
    assert tel['route_guard_parent_deficit_after'] == 20
    assert tel['route_guard_deficit_after'] <= 14
    assert not np.array_equal(action, expected)
    assert returned.base.defense.defender == -1 and returned.base.phase == 0
    assert returned.base.defense.last_turn == returned.base.last_turn == obs.timestep
    assert len(jax.tree.leaves(returned)) == 19
    assert all(x.shape == () and x.dtype == jnp.int32 for x in jax.tree.leaves(returned))


@pytest.mark.parametrize('name', ['unchanged-bound-build', 'productive-build', 'obsolete-screen-location'])
def test_real_negative_controls_keep_parent_action_and_memory(name):
    d, obs, key, memory = fixture(name)
    old = SentinelV10Agent(**RULES).step(obs, key, memory)
    np.testing.assert_array_equal(old[0], d['expected_parent_action'])
    action, returned, tel = SentinelV12Agent(**RULES).step(obs, key, memory)
    same_tree((action, returned), old[:2])
    assert not tel['route_guard_issued']


def test_captured_defender_is_released_instead_of_permanently_held():
    _, obs, key, memory = fixture('captured-defender')
    _, returned, tel = SentinelV12Agent(**RULES).step(obs, key, memory)
    assert returned.base.defense.defender == -1
    assert tel['commitment_observation_inconsistent']


def test_disabled_has_exact_parent_action_memory_and_telemetry():
    _, obs, key, memory = fixture('weakening-transfer')
    same_tree(SentinelV12Agent(**RULES, guard_routes=False).step(obs, key, memory),
              SentinelV10Agent(**RULES).step(obs, key, memory))


@pytest.mark.parametrize('target_army,split', [(3, 0), (20, 0), (9, 0), (8, 1)])
def test_static_attack_projection_matches_engine_even_without_capture(target_army, split):
    state = game.create_initial_state(jnp.zeros((5, 5), jnp.int32).at[4, 4].set(1).at[0, 0].set(2))
    ownership = state.ownership.at[0, 2, 2].set(True).at[1, 2, 3].set(True)
    state = state._replace(ownership=ownership,
        ownership_neutral=state.ownership_neutral.at[2, 2].set(False).at[2, 3].set(False),
        armies=state.armies.at[2, 2].set(10).at[2, 3].set(target_army))
    obs = game.get_full_observation(state, 0)
    action = jnp.array([0, 2, 2, 3, split], jnp.int32)
    actual = game.execute_action(state, 0, action)
    projected = _project_action(obs, action, _build_prices(obs.owned_cells & obs.generals))
    same_tree(projected, (actual.armies, actual.ownership[0], actual.ownership[1]))


def test_noncanonical_pass_and_build_directions_do_not_transfer_armies():
    _, obs, _, _ = fixture('unchanged-bound-build')
    prices = _build_prices(obs.owned_cells & (obs.generals | obs.castles))
    same_tree(_project_action(obs, jnp.array([1, 9, 16, 3, 1]), prices),
              (obs.armies, obs.owned_cells, obs.opponent_cells))
    armies, owned, enemies = _project_action(obs, jnp.array([2, 9, 16, 3, 1]), prices)
    np.testing.assert_array_equal(armies, obs.armies.at[9, 16].add(-45))
    same_tree((owned, enemies), (obs.owned_cells, obs.opponent_cells))


def test_deathtouch_general_capture_keeps_terminal_priority():
    state = game.create_initial_state(jnp.zeros((5, 5), jnp.int32).at[4, 4].set(1).at[2, 3].set(2))
    state = state._replace(ownership=state.ownership.at[0, 2, 2].set(True),
                          armies=state.armies.at[2, 2].set(2).at[2, 3].set(100), time=jnp.int32(800))
    obs = game.get_full_observation(state, 0)
    action = jnp.array([0, 2, 2, 3, 0])
    assert _winning(obs, action, 800) and not _winning(obs, action, None)
    agent = SentinelV12Agent(**RULES)
    chosen, _, tel = agent.step(obs, jax.random.PRNGKey(0), agent.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(chosen, action)
    assert not tel['route_guard_active'] and not tel['route_guard_issued']


def test_batched_guard_has_independent_fixed_memory_and_general_knowledge():
    _, a, ka, ma = fixture('weakening-transfer')
    _, b, kb, mb = fixture('unchanged-bound-build')
    agent = SentinelV12Agent(**RULES)
    parent = SentinelV10Agent(**RULES)
    batch = jax.tree.map(lambda x, y: jnp.stack((x, y)), a, b)
    memories = jax.tree.map(lambda x, y: jnp.stack((x, y)), ma, mb)
    actions, returned, tel = jax.jit(jax.vmap(agent.step))(batch, jnp.stack((ka, kb)), memories)
    np.testing.assert_array_equal(tel['route_guard_issued'], [True, False])
    for i, (obs, key, memory) in enumerate([(a, ka, ma), (b, kb, mb)]):
        old_action, old_memory, _ = parent.step(obs, key, memory)
        for name in ['enemy_general', 'last_turn', 'height', 'width']:
            np.testing.assert_array_equal(getattr(returned, name)[i], getattr(old_memory, name))
        if i == 1:
            same_tree((actions[i], jax.tree.map(lambda x: x[i], returned)), (old_action, old_memory))
    assert len(jax.tree.leaves(returned)) == 19
    assert all(x.shape == (2,) and x.dtype == jnp.int32 for x in jax.tree.leaves(returned))

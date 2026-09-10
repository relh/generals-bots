"""Capture-chain value, actual transport, public validity and parent isolation."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v16_agent import SentinelV16Agent
from generals.agents.sentinel_v17_agent import SentinelV17Agent, _best_chain, _capture_values
from generals.core import game
from tests.test_multiplayer import board as state_board
from tests.test_multiplayer import give
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import KEY, intercept_board, legal
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
PARENT = SentinelV16Agent(**RULES)
AGENT = SentinelV17Agent(**RULES)
DISABLED = SentinelV17Agent(**RULES, capture_chains=False)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)


def fixture(name):
    family = 'campaign_feeder' if name in ('juraj100', 'amin129', 'amin142') else 'capture_chain'
    data = json.loads((Path(__file__).parent / f'fixtures/{family}/{name}.json').read_text())
    obs = read_observation(io.StringIO(data['public_wire']), *data['shape'])
    memory = restore(PARENT.initial_memory(obs.armies.shape), data['incoming_memory'])
    return data, obs, memory, jnp.array(data['action_key'], jnp.uint32)


def recorded_action(tel, label):
    return jnp.array([tel[f'chain_{label}_{field}'] for field in ('kind', 'row', 'column', 'direction', 'split')])


@pytest.mark.parametrize('name,direction,residual,original', [('juraj109', 1, 10, 5), ('amin214', 0, 11, 6)])
def test_recorded_immediate_attack_selects_more_productive_same_packet_route(name, direction, residual, original):
    data, obs, memory, key = fixture(name)
    parent, _, _ = PARENT.step(obs, key, memory)
    action, returned, tel = AGENT.step(obs, key, memory)
    np.testing.assert_array_equal(parent, data['expected_current_action'])
    np.testing.assert_array_equal(recorded_action(tel, 'parent'), parent)
    assert action[3] == direction and np.array_equal(action[1:3], parent[1:3])
    assert tel['chain_started'] and tel['chain_override'] and not tel['chain_parent_action_issued']
    assert tel['chain_length'] == 3 and tel['chain_residual'] == residual
    assert tel['chain_comparison_available']
    assert tel['chain_original_length'] == 3 and tel['chain_original_residual'] == original
    assert tel['chain_home_safe'] and tel['actual_v8_action_issued'] and tel['mobilization_parent_action_issued']
    assert returned.base.phase == 4 and returned.base.remaining == 2
    assert returned.base.expires == obs.timestep + 2
    dest = int(tel['chain_target'])
    assert returned.base.expected_army == obs.armies[tuple(np.asarray(action[1:3]))] - 1 - obs.armies.reshape(-1)[dest]
    legal(obs, action)


@pytest.mark.parametrize('name', ['amin163', 'amin223', 'juraj100', 'amin129', 'amin142', 'amin250'])
def test_original_admissions_and_negative_first_actions_preserved(name):
    _, obs, memory, key = fixture(name)
    parent, parent_memory, parent_tel = PARENT.step(obs, key, memory)
    action, returned, tel = AGENT.step(obs, key, memory)
    np.testing.assert_array_equal(action, parent)
    assert not tel['chain_override'] and tel['chain_parent_action_issued']
    if name != 'amin223':
        assert not tel['chain_admission'] and not tel['chain_issued']
        assert not tel['chain_comparison_available']
        assert tel['chain_original_length'] == tel['chain_original_residual'] == -1
        same_tree(returned, parent_memory)
    else:
        assert tel['chain_started'] and tel['chain_length'] == 2
        assert tel['chain_residual'] == tel['chain_original_residual'] == 3
    same_tree({k: tel[k] for k in parent_tel}, parent_tel)


@pytest.mark.parametrize('name', ['juraj109', 'amin214', 'amin223', 'amin250', 'amin129'])
def test_disabled_complete_tuple_is_independent_frozen_v16(name):
    _, obs, memory, key = fixture(name)
    same_tree(DISABLED.step(obs, key, memory), PARENT.step(obs, key, memory))


def active_memory(obs, packet=(4, 2), first=(4, 3), final=(4, 4), expected=20):
    mem = AGENT.initial_memory(obs.armies.shape)
    w = obs.armies.shape[1]
    return mem._replace(last_turn=obs.timestep - 1, base=mem.base._replace(
        packet=jnp.int32(packet[0] * w + packet[1]), rally=jnp.int32(first[0] * w + first[1]),
        objective=jnp.int32(final[0] * w + final[1]), phase=jnp.int32(4), remaining=jnp.int32(2),
        expected_army=jnp.int32(expected), expires=obs.timestep + 1, last_turn=obs.timestep - 1,
    ))


def route_state():
    state = state_board({0: (8, 8), 1: (0, 0)}, size=9)
    for cell, army in [((8, 8), 30), ((4, 2), 20), ((3, 4), 1)]:
        state = give(state, 0, cell, army)
    for cell, army in [((4, 3), 2), ((4, 4), 1)]:
        state = give(state, 1, cell, army)
    return state._replace(time=jnp.int32(21))


def test_fixed_pending_route_executes_two_real_captures_and_clears_without_renewal():
    state = route_state()
    obs = game.get_observation(state, 0)
    memory = active_memory(obs)
    expiry = int(memory.base.expires)
    for column, force in [(2, 17), (3, 15)]:
        obs = game.get_observation(state, 0)
        action, memory, tel = AGENT.step(obs, KEY, memory)
        np.testing.assert_array_equal(action, [0, 4, column, 3, 0])
        assert tel['chain_continued'] and tel['chain_attack_issued']
        assert not tel['chain_comparison_available']
        assert tel['chain_original_length'] == tel['chain_original_residual'] == -1
        legal(obs, action)
        state, _ = game.step(state, jnp.stack((action, PASS)))
        assert state.ownership[0, 4, column + 1] and state.armies[4, column + 1] == force
        assert state.armies[4, column] == 1
        if column == 2:
            assert memory.base.remaining == 1 and memory.base.expires == expiry
            assert memory.base.rally == 4 * 9 + 4 and memory.base.expected_army == force
    assert memory.base.phase == 0 and tel['chain_finished']


@pytest.mark.parametrize('change', ['lost_packet', 'depleted', 'home_packet', 'lost_target', 'fog', 'mountain',
                                   'castle', 'counterforce', 'expired', 'gap', 'reset'])
def test_observed_disruption_aborts_fixed_chain_and_retains_current_parent_defense(change):
    obs = game.get_observation(route_state(), 0)
    memory = active_memory(obs)
    if change == 'lost_packet':
        obs = obs._replace(owned_cells=obs.owned_cells.at[4, 2].set(False))
    elif change == 'depleted':
        obs = obs._replace(armies=obs.armies.at[4, 2].set(19))
    elif change == 'home_packet':
        obs = obs._replace(generals=obs.generals.at[4, 2].set(True))
    elif change == 'lost_target':
        obs = obs._replace(owned_cells=obs.owned_cells.at[4, 3].set(True),
                           opponent_cells=obs.opponent_cells.at[4, 3].set(False))
    elif change == 'fog':
        obs = obs._replace(fog_cells=obs.fog_cells.at[4, 4].set(True))
    elif change == 'mountain':
        obs = obs._replace(mountains=obs.mountains.at[4, 4].set(True))
    elif change == 'castle':
        obs = obs._replace(castles=obs.castles.at[4, 4].set(True))
    elif change == 'counterforce':
        obs = obs._replace(opponent_cells=obs.opponent_cells.at[5, 4].set(True),
                           armies=obs.armies.at[5, 4].set(50))
    elif change == 'expired':
        memory = memory._replace(base=memory.base._replace(expires=obs.timestep - 1))
    elif change == 'gap':
        memory = memory._replace(last_turn=obs.timestep - 2)
    else:
        memory = memory._replace(last_turn=obs.timestep)
    _, returned, tel = AGENT.step(obs, KEY, memory)
    assert not tel['chain_issued'] and returned.base.phase != 4
    if change not in ('gap', 'reset'):
        assert tel['chain_released']
        assert any(bool(v) for k, v in tel.items() if k.startswith('chain_abort_'))


def test_projection_rejects_revisits_fog_zero_and_counts_captured_counterforce_once():
    obs = board([(5, 5, 30, True), (2, 1, 10, False)], [(2, 2, 2, False), (2, 3, 2, False)], shape=(6, 6))
    # Enemy2 on first target must not be charged again as a counterattacker.
    paths = jnp.array([[14, 15, 16], [14, 15, 14]], jnp.int32)
    valid, residual, _, _, _ = _capture_values(obs, 13, paths, jnp.array([3, 3]), jnp.ones((2, 3), bool))
    np.testing.assert_array_equal(valid, [True, False])
    assert residual[0] == 3
    fog = obs._replace(fog_cells=obs.fog_cells.at[2, 4].set(True))
    assert not _capture_values(fog, 13, paths[:1], jnp.array([3]), jnp.ones((1, 3), bool))[0][0]
    # First-action tie is retained even if deterministic direction order differs.
    symmetric = board([(5, 5, 30, True), (2, 2, 10, False)], [(2, 1, 1, False), (2, 3, 1, False)], shape=(6, 6))
    assert _best_chain(symmetric, 14, 3)[1] == 3


def test_active_chain_preserves_actual_build_and_winning_general():
    obs = game.get_observation(route_state(), 0)
    rich = obs._replace(armies=obs.armies.at[4, 2].set(80))
    action, returned, tel = AGENT.step(rich, KEY, active_memory(rich))
    assert action[0] == 2 and tel['chain_abort_priority'] and returned.base.phase != 4
    win = obs._replace(generals=obs.generals.at[4, 3].set(True))
    action, returned, tel = AGENT.step(win, KEY, active_memory(win))
    np.testing.assert_array_equal(action, [0, 4, 2, 3, 0])
    assert tel['chain_abort_priority'] and returned.base.phase != 4
    for neutral in (False, True):
        city = obs._replace(castles=obs.castles.at[4, 3].set(True))
        if neutral:
            city = city._replace(opponent_cells=city.opponent_cells.at[4, 3].set(False),
                                 neutral_cells=city.neutral_cells.at[4, 3].set(True))
        action, returned, tel = AGENT.step(city, KEY, active_memory(city))
        np.testing.assert_array_equal(action, [0, 4, 2, 3, 0])
        assert tel['chain_abort_priority'] and not tel['chain_issued']
        assert returned.base.phase != 4
        legal(city, action)


def test_fresh_defense_and_outer_pursuit_preserve_actual_reservations_and_masks():
    obs = intercept_board()
    memory = active_memory(obs, packet=(2, 2), first=(1, 2), final=(1, 3))
    action, returned, tel = AGENT.step(obs, KEY, memory)
    blank = PARENT.initial_memory(obs.armies.shape)._replace(last_turn=memory.last_turn)
    expected, expected_memory, _ = PARENT.step(obs, KEY, blank)
    np.testing.assert_array_equal(action, expected)
    same_tree(returned.base.defense, expected_memory.base.defense)
    assert not tel['chain_issued'] and returned.base.phase != 4
    obs = game.get_observation(route_state(), 0)
    memory = active_memory(obs)._replace(enemy_general=jnp.int32(1))
    obs = obs._replace(fog_cells=obs.fog_cells.at[0, 1].set(True))
    _, returned, tel = AGENT.step(obs, KEY, memory)
    assert tel['chain_issued'] and tel['pursuit_override'] and not tel['actual_v8_action_issued']
    assert returned.base.phase == 0


def test_jit_vmap_fixed_native19_and_batch_plan_isolation():
    obs = game.get_observation(route_state(), 0)
    invalid = obs._replace(fog_cells=obs.fog_cells.at[4, 4].set(True))
    batch = jax.tree.map(lambda a, b: None if a is None else jnp.stack((a, b)), obs, invalid)
    memory = active_memory(obs)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    _, returned, tel = jax.jit(jax.vmap(AGENT.step))(batch, jnp.stack((KEY, KEY)), memories)
    np.testing.assert_array_equal(tel['chain_issued'], [True, False])
    leaves = jax.tree.leaves(returned)
    assert len(leaves) == 19 and all(x.dtype == jnp.int32 and x.shape == (2,) for x in leaves)

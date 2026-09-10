"""Bounded defender retention: original full tuples and independently specified cases."""

import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v6_agent import SentinelV6Agent, DefenderMemory
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v19_agent import SentinelV19Agent, _RetainedDefender
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
ENABLED = SentinelV19Agent(**RULES)
DISABLED = SentinelV19Agent(**RULES, retain_defense=False)
V10V6 = SentinelV10Agent(**RULES, parent_version=6)
INNER = _RetainedDefender(V10V6.base)
KEY = jax.random.PRNGKey(0)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)
V18_NAMES = ['v18_amin245', 'v18_amin347', 'v18_amin348', 'v18_amin349', 'v18_amin427',
             *[f'v18_juraj{t}' for t in range(438, 443)], 'v18_mybot66']


def archived(name, agent=DISABLED):
    path = Path(__file__).parent / 'fixtures/retained_defense' / (name + '.json')
    data = json.loads(path.read_text())
    assert hashlib.sha256(data['public_wire'].encode()).hexdigest() == data['wire_sha256']
    obs = read_observation(io.StringIO(data['public_wire']), *data['shape'])
    memory = restore(agent.initial_memory(data['shape']), data['incoming_memory'])
    return data, obs, memory, jnp.array(data['action_key'], jnp.uint32)


def assert_archived(result, data):
    action, memory, telemetry = result
    np.testing.assert_array_equal(action, data['expected_parent_action'])
    same_tree(memory, restore(memory, data['expected_parent_memory']))
    assert telemetry.keys() == data['expected_parent_telemetry'].keys()
    for name, expected in data['expected_parent_telemetry'].items():
        np.testing.assert_array_equal(telemetry[name], expected)


@pytest.mark.parametrize('name', V18_NAMES)
def test_disabled_exact_recorded_v18_full_tuple(name):
    data, obs, memory, key = archived(name)
    result = DISABLED.step(obs, key, memory)
    assert_archived(result, data)
    leaves = jax.tree.leaves(result[1])
    assert len(leaves) == 19
    assert all(np.asarray(x).dtype == np.int32 and np.asarray(x).shape == () for x in leaves)


@pytest.mark.parametrize('name,eligible', [
    ('v18_amin245', False), ('v18_amin347', False), ('v18_amin348', True),
    ('v18_amin349', False), ('v18_amin427', False), ('v18_juraj438', False),
    ('v18_juraj439', True), ('v18_juraj440', False), ('v18_juraj441', False),
    ('v18_juraj442', False),
])
def test_original_inputs_enable_only_existing_valid_defender(name, eligible):
    data, obs, memory, key = archived(name)
    action, returned, tel = ENABLED.step(obs, key, memory)
    assert bool(tel['retained_eligible']) == eligible
    assert bool(tel['retained_action_issued']) == (eligible and bool(tel['retained_outer_action_issued']))
    assert len(jax.tree.leaves(returned)) == 19
    if eligible:
        assert returned.base.defense.expires == memory.base.defense.expires
        assert not tel['retained_visible_need']
        assert tel['commitment_continued'] or tel['commitment_held']
    if name == 'v18_amin348':
        np.testing.assert_array_equal(action, [0, 8, 13, 3, 0])
        assert returned.base.defense.defender == 158 and returned.base.defense.target == 161
        assert returned.base.defense.distance == 3 and returned.base.defense.expires == 357
        assert tel['retained_action_override']
    if name == 'v18_juraj439':
        np.testing.assert_array_equal(action, data['expected_parent_action'])
        assert tel['retained_parent_progresses'] and not tel['retained_inner_override']
        assert returned.base.defense.defender == returned.base.defense.target == 129
        assert returned.base.defense.expires == 448
    if name == 'v18_amin427':
        assert not tel['retained_consistent'] and tel['retained_priority']
        assert tel['retained_parent_adjacent_threat'] == 11
    if name == 'v18_amin245':
        # Recorded defender already occupies home; it is not a reusable field screen.
        assert not tel['retained_consistent']


@pytest.mark.parametrize('turn', [477, 805, 809, 923, 947, 948])
def test_successful_control_archived_outer_and_real_inner_priority(turn):
    data, obs, memory, key = archived(f'v10_v6_juraj{turn}', V10V6)
    assert_archived(V10V6.step(obs, key, memory), data)
    # The actual inner V6 tuple was NOT archived. Obtain it explicitly here;
    # comparisons below are fresh unit inference, not fabricated historical output.
    parent = V10V6.base.step(obs, key, memory)
    action, returned, tel = INNER.step(obs, key, memory)
    same_tree((action, returned), parent[:2])
    assert not tel['retained_eligible']
    if turn in (805, 923):
        assert not tel['retained_unexpired']
    if turn == 809:
        assert tel['retained_selected_build']
        np.testing.assert_array_equal(action, [2, 7, 15, 0, 0])
    if turn == 947:
        assert tel['retained_priority'] and tel['retained_parent_intercept_guard']
    if turn == 948:
        assert tel['retained_selected_nonowned_move']
        np.testing.assert_array_equal(action, [0, 6, 17, 0, 0])
    if turn == 477:
        # A valid no-need route exists, but the already selected neutral capture has priority.
        assert tel['retained_valid'] and tel['retained_route_valid']
        assert not tel['retained_visible_need'] and tel['retained_selected_nonowned_move']
        np.testing.assert_array_equal(action, [0, 10, 9, 0, 0])
    for field, value in parent[2].items():
        np.testing.assert_array_equal(tel['retained_parent_' + field], value)


class FixedV6:
    """Declared V6 proposal isolates retention arbitration, not parent policy strength."""

    def __init__(self, action=PASS, **flags):
        self.action = jnp.asarray(action, jnp.int32)
        self.deathtouch_turn = 800
        self.base = SentinelV6Agent(**RULES).base
        names = ['adjacent_threat', 'intercept_guard', 'commitment_started', 'building',
                 'commitment_continued', 'commitment_held', 'commitment_released',
                 'commitment_observation_inconsistent', 'commitment_expired',
                 'commitment_observation_gap', 'commitment_reverse_blocked',
                 'commitment_override', 'commitment_remaining', 'commitment_required',
                 'commitment_arrival_army']
        self.telemetry = {name: jnp.array(False) for name in names}
        for name in ['commitment_remaining', 'commitment_required', 'commitment_arrival_army']:
            self.telemetry[name] = jnp.float32(0)
        self.telemetry.update({name: jnp.asarray(value) for name, value in flags.items()})

    def initial_memory(self, shape):
        return DefenderMemory(*[jnp.int32(x) for x in [-1, -1, -1, 0, -1, -1, 0]])

    def step(self, obs, key, memory):
        return self.action, self.initial_memory(obs.armies.shape)._replace(last_turn=obs.timestep), self.telemetry


def transit_case():
    obs = board([(0, 0, 20, True), (2, 2, 10, False), (1, 2, 2, False),
                 (2, 1, 3, False), (1, 1, 1, False)], shape=(4, 4), time=10)
    memory = DefenderMemory(*map(jnp.int32, [10, 5, 14, 2, 13, 9, 10]))
    return obs, memory


@pytest.mark.parametrize('parent,expected', [
    ([1, 0, 0, 0, 0], [0, 2, 2, 0, 0]),
    ([0, 2, 2, 2, 0], [0, 2, 2, 2, 0]),
    ([0, 2, 2, 2, 1], [0, 2, 2, 0, 0]),
])
def test_cardinal_tie_order_preserves_only_original_full_progress(parent, expected):
    obs, memory = transit_case()
    action, returned, tel = _RetainedDefender(FixedV6(parent)).step(obs, KEY, memory)
    np.testing.assert_array_equal(action, expected)
    assert tel['retained_eligible'] and tel['retained_continued']
    assert returned.distance == 1 and returned.expires == 13
    assert returned.expected_army == (12 if expected[3] == 2 else 11)
    assert tel['retained_expected_army'] == returned.expected_army
    # Inherited target-arrival telemetry retains the original V6 estimate.
    assert tel['commitment_arrival_army'] == tel['retained_parent_commitment_arrival_army'] == 0


def test_repeated_field_hold_never_refreshes_original_expiry():
    obs, memory = transit_case()
    memory = memory._replace(target=memory.defender, distance=jnp.int32(0))
    agent = _RetainedDefender(FixedV6())
    for turn in range(10, 14):
        action, memory, tel = agent.step(obs._replace(timestep=jnp.int32(turn)), KEY, memory)
        assert tel['retained_held'] and memory.expires == 13 and memory.last_turn == turn
        np.testing.assert_array_equal(action, PASS)
    _, returned, tel = agent.step(obs._replace(timestep=jnp.int32(14)), KEY, memory)
    assert not tel['retained_eligible'] and not tel['retained_unexpired']
    assert returned.defender == -1


def test_home_arrival_clears_all_defender_fields_immediately():
    obs = board([(0, 0, 20, True), (1, 0, 9, False)], shape=(4, 4), time=10)
    memory = DefenderMemory(*map(jnp.int32, [4, 0, 8, 1, 13, 9, 9]))
    agent = _RetainedDefender(FixedV6())
    action, returned, tel = agent.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, [0, 1, 0, 0, 0])
    same_tree(returned, agent.initial_memory(obs.armies.shape)._replace(last_turn=jnp.int32(10)))
    assert tel['retained_finished'] and tel['commitment_released']


@pytest.mark.parametrize('reason', ['negative', 'out_of_range', 'target_out_of_range', 'expired', 'gap',
                                    'unowned_source', 'unowned_target', 'army_shortfall', 'one_army',
                                    'route_longer', 'missing_home'])
def test_invalid_inconsistent_or_stale_memory_gets_no_grace(reason):
    obs, memory = transit_case()
    if reason == 'negative': memory = memory._replace(defender=jnp.int32(-1))
    elif reason == 'out_of_range': memory = memory._replace(defender=jnp.int32(16))
    elif reason == 'target_out_of_range': memory = memory._replace(target=jnp.int32(16))
    elif reason == 'expired': memory = memory._replace(expires=jnp.int32(9))
    elif reason == 'gap': memory = memory._replace(last_turn=jnp.int32(8))
    elif reason == 'unowned_source': obs = obs._replace(owned_cells=obs.owned_cells.at[2, 2].set(False))
    elif reason == 'unowned_target': obs = obs._replace(owned_cells=obs.owned_cells.at[1, 1].set(False))
    elif reason == 'army_shortfall': memory = memory._replace(expected_army=jnp.int32(11))
    elif reason == 'one_army': obs = obs._replace(armies=obs.armies.at[2, 2].set(1))
    elif reason == 'route_longer': memory = memory._replace(distance=jnp.int32(1))
    elif reason == 'missing_home': obs = obs._replace(generals=jnp.zeros_like(obs.generals))
    action, returned, tel = _RetainedDefender(FixedV6()).step(obs, KEY, memory)
    assert not tel['retained_eligible']
    np.testing.assert_array_equal(action, PASS)
    assert returned.defender == -1


@pytest.mark.parametrize('reason', ['build', 'neutral_capture', 'enemy_attack', 'guard', 'adjacent', 'new_plan'])
def test_selected_priority_is_never_replaced(reason):
    obs, memory = transit_case()
    flags = {}
    if reason == 'build': action = [2, 2, 2, 0, 0]
    elif reason in ('neutral_capture', 'enemy_attack'):
        action = [0, 2, 2, 3, 0]
        if reason == 'enemy_attack':
            obs = obs._replace(opponent_cells=obs.opponent_cells.at[2, 3].set(True),
                               neutral_cells=obs.neutral_cells.at[2, 3].set(False),
                               armies=obs.armies.at[2, 3].set(1))
    else:
        action = PASS
        flags[{'guard': 'intercept_guard', 'adjacent': 'adjacent_threat', 'new_plan': 'commitment_started'}[reason]] = True
    selected, returned, tel = _RetainedDefender(FixedV6(action, **flags)).step(obs, KEY, memory)
    np.testing.assert_array_equal(selected, action)
    assert not tel['retained_eligible']
    assert returned.defender == -1


def test_field_hold_uses_profitable_other_source_and_preserves_selected_elsewhere_move():
    obs = board([(0, 0, 20, True), (2, 2, 10, False), (2, 1, 1, False),
                 (0, 2, 20, False)], shape=(4, 4), time=10)
    memory = DefenderMemory(*map(jnp.int32, [10, 10, 14, 0, 13, 9, 10]))
    action, returned, tel = _RetainedDefender(FixedV6([0, 2, 2, 2, 0])).step(obs, KEY, memory)
    assert tel['retained_held'] and tel['retained_inner_override']
    assert action[0] == 0 and tuple(map(int, action[1:3])) != (2, 2)
    assert returned.defender == 10 and returned.expected_army == 10
    elsewhere = jnp.array([0, 0, 2, 2, 0], jnp.int32)
    # Make the selected alternative an owned transfer; nonowned moves have priority anyway.
    obs = obs._replace(owned_cells=obs.owned_cells.at[0, 1].set(True),
                       neutral_cells=obs.neutral_cells.at[0, 1].set(False),
                       armies=obs.armies.at[0, 1].set(1))
    chosen, _, tel = _RetainedDefender(FixedV6(elsewhere)).step(obs, KEY, memory)
    np.testing.assert_array_equal(chosen, elsewhere)
    assert tel['retained_held'] and not tel['retained_inner_override']


@pytest.mark.parametrize('blocked_layer', ['none', 'rear', 'mobilization', 'v8', 'offense'])
def test_outer_selectors_mask_inner_issued_events(blocked_layer):
    obs, _ = transit_case()
    agent = SentinelV19Agent(**RULES)
    memory = agent.initial_memory(obs.armies.shape)
    tel = {'rear_parent_action_issued': blocked_layer != 'rear',
           'mobilization_parent_action_issued': blocked_layer != 'mobilization',
           'actual_v8_action_issued': blocked_layer != 'v8',
           'offense_override': blocked_layer == 'offense',
           'retained_inner_issued': True, 'retained_inner_override': True}
    # Only the final attribution boundary is isolated here; real layered actions
    # and nineteen-field memory are tested using the archived fixtures above.
    class FixedOuter:
        def step(self, obs, key, before):
            return PASS, before, {k: jnp.asarray(v) for k, v in tel.items()}
    agent.parent = FixedOuter()
    action, returned, result = agent.step(obs, KEY, memory)
    same_tree(returned, memory)
    np.testing.assert_array_equal(action, PASS)
    assert bool(result['retained_action_issued']) == (blocked_layer == 'none')
    assert bool(result['retained_action_override']) == (blocked_layer == 'none')


def test_current_visible_need_prevents_no_need_grace():
    obs, memory = transit_case()
    obs = obs._replace(opponent_cells=obs.opponent_cells.at[0, 3].set(True),
                       neutral_cells=obs.neutral_cells.at[0, 3].set(False),
                       armies=obs.armies.at[0, 3].set(100))
    action, returned, tel = _RetainedDefender(FixedV6()).step(obs, KEY, memory)
    assert tel['retained_valid'] and tel['retained_visible_need']
    assert not tel['retained_eligible'] and returned.defender == -1
    np.testing.assert_array_equal(action, PASS)


def test_actual_unchanged_juraj439_carry_respects_selected_enemy_capture_at440():
    data, obs, memory, key = archived('v18_juraj439')
    action, retained, tel = ENABLED.step(obs, key, memory)
    # Same actual action means the next original public observation is a valid
    # saved-input continuation; this is not a changed-game outcome claim.
    np.testing.assert_array_equal(action, data['expected_parent_action'])
    assert retained.base.defense.defender == retained.base.defense.target == 129
    _, next_obs, _, next_key = archived('v18_juraj440')
    held_action, held_memory, held_tel = ENABLED.step(next_obs, next_key, retained)
    assert held_tel['retained_valid'] and held_tel['retained_arrived']
    assert not held_tel['retained_visible_need']
    assert held_tel['retained_selected_nonowned_move'] and held_tel['retained_priority']
    assert not held_tel['retained_held'] and not held_tel['retained_action_issued']
    assert held_memory.base.defense.defender == -1
    # The real inner parent captures enemy1 to the left; it is preserved even
    # though the original outer action without retained memory went up.
    assert next_obs.opponent_cells[6, 14] and next_obs.armies[6, 14] == 1
    np.testing.assert_array_equal(held_action, [0, 6, 15, 2, 0])


def test_actual_engine_convoy_survives_growth_then_clears_on_home_arrival():
    from generals.core import game
    from tests.test_multiplayer import board as state_board, give

    state = state_board({0: (0, 0), 1: (5, 5)}, size=6)
    for cell, army in [((0, 0), 20), ((1, 0), 2), ((2, 0), 10)]:
        state = give(state, 0, cell, army)
    state = state._replace(time=jnp.int32(49))
    memory = DefenderMemory(*map(jnp.int32, [12, 0, 18, 2, 52, 48, 10]))
    agent = _RetainedDefender(FixedV6())
    action, memory, tel = agent.step(game.get_observation(state, 0), KEY, memory)
    np.testing.assert_array_equal(action, [0, 2, 0, 0, 0])
    assert memory.defender == 6 and memory.expected_army == 11 and memory.expires == 52
    state, _ = game.step(state, jnp.stack((action, PASS)))
    assert state.time == 50 and state.armies[1, 0] == 12
    assert state.armies[2, 0] == 2 and state.armies[0, 0] == 22
    # Next observed army exceeds the exact pre-growth memory floor. The real
    # transition must still admit continuation, without extending expiry.
    action, returned, tel = agent.step(game.get_observation(state, 0), KEY, memory)
    assert tel['retained_consistent'] and tel['retained_finished']
    assert tel['retained_expected_army'] == 33
    np.testing.assert_array_equal(action, [0, 1, 0, 0, 0])
    state, _ = game.step(state, jnp.stack((action, PASS)))
    assert state.time == 51 and state.armies[0, 0] == 33 and state.armies[1, 0] == 1
    same_tree(returned, agent.initial_memory((6, 6))._replace(last_turn=jnp.int32(50)))

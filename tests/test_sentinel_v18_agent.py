"""Rear expansion preserves frozen parents and pays real action/production costs."""

import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_agent import _distance
from generals.agents.sentinel_v18_agent import SentinelV18Agent, _rear_proposal
from generals.core import game
from tests.test_multiplayer import board as state_board
from tests.test_multiplayer import give
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
AGENT = SentinelV18Agent(**RULES)
DISABLED = SentinelV18Agent(**RULES, rear_expansion=False)
KEY = jax.random.PRNGKey(0)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)
TRANSFER = jnp.array([0, 4, 0, 3, 0], jnp.int32)


def archived(name):
    path = Path(__file__).parent / 'fixtures/rear_expansion' / (name + '.json')
    data = json.loads(path.read_text())
    assert hashlib.sha256(data['public_wire'].encode()).hexdigest() == data['wire_sha256']
    obs = read_observation(io.StringIO(data['public_wire']), *data['shape'])
    memory = restore(AGENT.initial_memory(data['shape']), data['incoming_memory'])
    return data, obs, memory, jnp.array(data['action_key'], jnp.uint32)


@pytest.mark.parametrize('name,issued', [
    ('amin58', False), ('amin129', False), ('amin169', False),
    ('amin150', True), ('juraj78', True), ('mybot64', False),
])
def test_archived_disabled_full_tuple_and_enabled_attribution(name, issued):
    data, obs, memory, key = archived(name)
    parent_action, parent_memory, parent_tel = DISABLED.step(obs, key, memory)
    np.testing.assert_array_equal(parent_action, data['expected_parent_action'])
    same_tree(parent_memory, restore(parent_memory, data['expected_parent_memory']))
    assert parent_tel.keys() == data['expected_parent_telemetry'].keys()
    for field, value in data['expected_parent_telemetry'].items():
        np.testing.assert_array_equal(parent_tel[field], value)
    action, returned, tel = AGENT.step(obs, key, memory)
    same_tree(returned, parent_memory)
    assert len(jax.tree.leaves(returned)) == 19
    for field, value in parent_tel.items():
        np.testing.assert_array_equal(tel[field], value)
    assert bool(tel['rear_issued']) == issued
    assert bool(tel['rear_parent_action_issued']) == (not issued)
    assert bool(jnp.any(action != parent_action)) == issued
    if issued:
        assert tel['rear_eligible'] and tel['rear_home_safe'] and tel['rear_safety_checked']
        r, c = map(int, action[1:3])
        assert obs.armies[r, c] == 2 and obs.owned_cells[r, c]
        assert not obs.generals[r, c] and not obs.castles[r, c]
        assert (r, c) != tuple(map(int, parent_action[1:3]))
        assert memory.base.phase == returned.base.phase == 0
        assert memory.base.defense.defender == returned.base.defense.defender == -1
    else:
        np.testing.assert_array_equal(action, parent_action)
        assert not tel['rear_safety_checked']
    if name == 'mybot64':
        assert not tel['rear_parent_owned_transfer']
    if name == 'amin58':
        assert not tel['rear_visible_contact']
    if name == 'amin129':
        assert tel['rear_returned_offense']
    if name == 'amin169':
        assert tel['rear_defense_priority'] and tel['rear_incoming_offense']


def proposal_board():
    return board([(0, 0, 20, True), (2, 2, 2, False), (4, 0, 20, False), (4, 1, 1, False)],
                 shape=(5, 5), time=49)


def propose(obs, parent=TRANSFER):
    distance = _distance(~(obs.mountains | obs.structures_in_fog), obs.owned_cells & obs.generals)
    return _rear_proposal(obs, parent, distance)


def test_proposal_uses_home_distance_then_branches_then_stable_direction_order():
    obs = proposal_board()
    action, available, count, branches = propose(obs)
    assert available and count == 4 and branches == 3
    # Up and left both have distance3 and three neutral exits; flat direction order selects up.
    np.testing.assert_array_equal(action, [0, 2, 2, 0, 0])
    fewer_up_branches = obs._replace(neutral_cells=obs.neutral_cells.at[0, 2].set(False))
    action, _, _, branches = propose(fewer_up_branches)
    np.testing.assert_array_equal(action, [0, 2, 2, 2, 0])
    assert branches == 3
    # Move home two columns: up is one step away versus left's three, despite fewer exits.
    near_up = fewer_up_branches._replace(
        generals=obs.generals.at[0, 0].set(False).at[0, 2].set(True),
        owned_cells=obs.owned_cells.at[0, 2].set(True),
        neutral_cells=fewer_up_branches.neutral_cells.at[0, 2].set(False),
    )
    np.testing.assert_array_equal(propose(near_up)[0], [0, 2, 2, 0, 0])


@pytest.mark.parametrize('reason', ['one', 'three', 'castle', 'general', 'parent_source', 'hostile_source'])
def test_proposal_rejects_ineligible_donor(reason):
    obs = proposal_board()
    parent = TRANSFER
    if reason in ('one', 'three'):
        obs = obs._replace(armies=obs.armies.at[2, 2].set(1 if reason == 'one' else 3))
    elif reason == 'castle':
        obs = obs._replace(castles=obs.castles.at[2, 2].set(True))
    elif reason == 'general':
        obs = obs._replace(generals=obs.generals.at[2, 2].set(True))
    elif reason == 'parent_source':
        parent = jnp.array([0, 2, 2, 0, 0], jnp.int32)
    else:
        obs = obs._replace(opponent_cells=obs.opponent_cells.at[2, 3].set(True),
                           neutral_cells=obs.neutral_cells.at[2, 3].set(False),
                           armies=obs.armies.at[2, 3].set(2))
    assert not propose(obs, parent)[1]


@pytest.mark.parametrize('reason', ['fog', 'structure_fog', 'castle', 'mountain', 'defended', 'hostile_target'])
def test_proposal_rejects_nonempty_or_unsafe_target(reason):
    obs = proposal_board()
    only = jnp.zeros_like(obs.neutral_cells).at[1, 2].set(True)
    obs = obs._replace(neutral_cells=only)
    if reason == 'fog':
        obs = obs._replace(fog_cells=obs.fog_cells.at[1, 2].set(True))
    elif reason == 'structure_fog':
        obs = obs._replace(structures_in_fog=obs.structures_in_fog.at[1, 2].set(True))
    elif reason == 'castle':
        obs = obs._replace(castles=obs.castles.at[1, 2].set(True))
    elif reason == 'mountain':
        obs = obs._replace(mountains=obs.mountains.at[1, 2].set(True))
    elif reason == 'defended':
        obs = obs._replace(armies=obs.armies.at[1, 2].set(1))
    else:
        obs = obs._replace(opponent_cells=obs.opponent_cells.at[0, 2].set(True),
                           armies=obs.armies.at[0, 2].set(2))
    assert not propose(obs)[1]


class FixedParent:
    """A declared parent decision isolates arbitration from campaign selection."""

    def __init__(self, action, returned, **flags):
        self.action, self.returned = action, returned
        names = ('remembered_general_available', 'offense_defense_priority', 'intercept_override',
                 'intercept_guard', 'commitment_started', 'commitment_continued', 'commitment_held',
                 'commitment_reverse_blocked', 'commitment_override', 'pursuit_issued', 'mobilization_issued')
        self.telemetry = {name: jnp.bool_(False) for name in names}
        self.telemetry.update(intercept_home_deficit=jnp.float32(0), adjacent_threat=jnp.float32(0),
                              general_reserve=jnp.float32(3), general_army=jnp.int32(20),
                              inherited_probe=jnp.int32(713))
        self.telemetry.update({name: jnp.asarray(value) for name, value in flags.items()})

    def step(self, obs, key, memory):
        return self.action, self.returned, self.telemetry


def isolated(obs, action=TRANSFER, incoming=None, returned=None, **flags):
    agent = SentinelV18Agent(**RULES)
    incoming = agent.initial_memory(obs.armies.shape) if incoming is None else incoming
    returned = incoming._replace(last_turn=obs.timestep) if returned is None else returned
    agent.parent = FixedParent(action, returned, **flags)
    return agent, incoming, returned


def contacted_board():
    obs = proposal_board()
    return obs._replace(opponent_cells=obs.opponent_cells.at[4, 4].set(True),
                        neutral_cells=obs.neutral_cells.at[4, 4].set(False),
                        armies=obs.armies.at[4, 4].set(1))


@pytest.mark.parametrize('priority', ['incoming_offense', 'returned_offense', 'incoming_defender',
                                      'returned_defender', 'intercept_guard', 'commitment_held',
                                      'pursuit_issued', 'mobilization_issued', 'remembered_general', 'visible_general'])
def test_arbitration_preserves_priority_parent_memory_and_events(priority):
    obs = contacted_board()
    agent, incoming, returned = isolated(obs)
    flags = {}
    if priority.endswith('offense'):
        memory = incoming if priority.startswith('incoming') else returned
        memory = memory._replace(base=memory.base._replace(phase=jnp.int32(1)))
        if priority.startswith('incoming'):
            incoming = memory
        else:
            returned = memory
    elif priority.endswith('defender'):
        memory = incoming if priority.startswith('incoming') else returned
        memory = memory._replace(base=memory.base._replace(defense=memory.base.defense._replace(defender=jnp.int32(22))))
        if priority.startswith('incoming'):
            incoming = memory
        else:
            returned = memory
    elif priority == 'visible_general':
        obs = obs._replace(generals=obs.generals.at[4, 4].set(True))
    elif priority == 'remembered_general':
        incoming = incoming._replace(enemy_general=jnp.int32(24))
    else:
        flags[priority] = True
    agent.parent = FixedParent(TRANSFER, returned, **flags)
    action, actual_memory, tel = agent.step(obs, KEY, incoming)
    np.testing.assert_array_equal(action, TRANSFER)
    same_tree(actual_memory, returned)
    assert tel['rear_available'] and tel['rear_priority'] and not tel['rear_issued']
    assert tel['rear_parent_action_issued'] and not tel['rear_safety_checked']
    assert tel['inherited_probe'] == 713


@pytest.mark.parametrize('kind', [1, 2])
def test_pass_or_build_is_never_replaced(kind):
    obs = contacted_board()
    parent = jnp.array([kind, 4, 0, 0, 0], jnp.int32)
    agent, incoming, returned = isolated(obs, parent)
    action, actual_memory, tel = agent.step(obs, KEY, incoming)
    np.testing.assert_array_equal(action, parent)
    same_tree(actual_memory, returned)
    assert not tel['rear_parent_owned_transfer'] and not tel['rear_issued']


def test_actual_capture_preserves_main_packet_and_earns_next_land_tick():
    state = state_board({0: (0, 0), 1: (5, 5)}, size=6)
    for cell, army in [((0, 0), 20), ((2, 2), 2), ((4, 0), 20), ((4, 1), 1)]:
        state = give(state, 0, cell, army)
    state = give(state, 1, (4, 2), 1)._replace(time=jnp.int32(48))
    obs = game.get_observation(state, 0)
    agent, memory, returned = isolated(obs)
    action, actual_memory, tel = agent.step(obs, KEY, memory)
    assert tel['rear_issued'] and not tel['rear_parent_action_issued']
    same_tree(actual_memory, returned)
    source, target = int(tel['rear_source']), int(tel['rear_target'])
    assert state.armies.reshape(-1)[source] == 2
    captured, _ = game.step(state, jnp.stack((action, PASS)))
    assert captured.ownership[0].reshape(-1)[target]
    assert captured.armies.reshape(-1)[source] == captured.armies.reshape(-1)[target] == 1
    assert captured.armies[4, 0] == 20 and captured.armies[4, 1] == 1
    # The transport packet was not spent, but its original move did not execute.
    baseline, _ = game.step(state, jnp.stack((TRANSFER, PASS)))
    assert baseline.armies[4, 0] == 1 and baseline.armies[4, 1] == 20
    grown, _ = game.step(captured, jnp.stack((PASS, PASS)))
    baseline_grown, _ = game.step(baseline, jnp.stack((PASS, PASS)))
    assert grown.time == 50 and grown.armies.reshape(-1)[target] == 2
    assert game.get_info(grown).army[0] == game.get_info(baseline_grown).army[0] + 1


def test_proposal_tie_uses_first_source_and_rejects_disconnected_targets():
    obs = board([(0, 0, 20, True), (0, 2, 2, False), (1, 1, 2, False),
                 (4, 0, 20, False), (4, 1, 1, False)], shape=(5, 5))
    obs = obs._replace(neutral_cells=jnp.zeros_like(obs.neutral_cells).at[0, 1].set(True))
    action, available, count, branches = propose(obs)
    assert available and count == 2 and branches == 0
    np.testing.assert_array_equal(action, [0, 0, 2, 2, 0])
    blocked = proposal_board()
    blocked = blocked._replace(mountains=blocked.mountains.at[0, 1].set(True).at[1, 0].set(True))
    assert not propose(blocked)[1]


def test_selected_proposal_home_failure_keeps_parent_without_trying_safe_runner_up():
    from generals.agents.sentinel_v7_agent import _preserves_home

    allowed = {(1, 0), (1, 1), (1, 2), (1, 3), (0, 1), (2, 0), (3, 0), (3, 1),
               (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)}
    obs = board([(1, 0, 3, True), (1, 1, 2, False), (2, 0, 1, False),
                 (3, 0, 2, False), (4, 3, 20, False), (4, 4, 1, False)],
                [(1, 3, 9, False)], mountains=[(r, c) for r in range(5) for c in range(5)
                                            if (r, c) not in allowed], shape=(5, 5), time=100)
    parent = jnp.array([0, 4, 3, 3, 0], jnp.int32)
    agent, memory, returned = isolated(obs, parent, general_army=3)
    action, actual_memory, tel = agent.step(obs, KEY, memory)
    # The only enemy route is (1,3)->(1,2)->(1,1)->home. Public route cost is
    # (0+1)+(2+1)+(3+1)+one home growth =9, falling to8 if donor(1,1) spends1.
    # The nearby branch(0,1) wins ranking; a separate farther donor has safe options.
    assert tel['rear_candidate_count'] == 3
    assert tel['rear_source'] == 6 and tel['rear_target'] == 1
    assert tel['rear_eligible'] and tel['rear_safety_checked'] and not tel['rear_home_safe']
    assert not tel['rear_issued'] and tel['rear_parent_action_issued']
    np.testing.assert_array_equal(action, parent)
    same_tree(actual_memory, returned)
    terrain = ~obs.mountains
    home = obs.owned_cells & obs.generals
    distance = _distance(terrain, home)
    assert distance[0, 1] == 2 and distance[3, 1] == 3
    safe_runner_up = jnp.array([0, 3, 0, 3, 0], jnp.int32)
    assert _preserves_home(obs, safe_runner_up, terrain, home, distance, 800)


def test_missing_home_preserves_parent_and_skips_safety_check():
    obs = contacted_board()
    obs = obs._replace(generals=jnp.zeros_like(obs.generals))
    agent, memory, returned = isolated(obs)
    action, actual_memory, tel = agent.step(obs, KEY, memory)
    assert not tel['rear_has_home'] and not tel['rear_eligible']
    assert not tel['rear_safety_checked'] and not tel['rear_issued']
    assert tel['rear_source'] == tel['rear_target'] == -1
    np.testing.assert_array_equal(action, TRANSFER)
    same_tree(actual_memory, returned)

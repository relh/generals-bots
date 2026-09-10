"""Original control preservation; these saved frames do not prove game strength."""

import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v20_agent import SentinelV20Agent
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

FIXTURES = Path(__file__).parent / 'fixtures' / 'branch_collection'
V10_NAMES = [f'v10-amin-{t}' for t in (348, 352, 359, 362)] + [f'v10-juraj-{t}' for t in (348, 420)]
LEGACY_NAMES = [f'v10-v6-juraj-{t}' for t in (477, 805, 948)]
RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
PARENT = SentinelV10Agent(**RULES)
LEGACY = SentinelV10Agent(**RULES, parent_version=6)
DISABLED = SentinelV20Agent(**RULES, branch_collection=False)
ENABLED = SentinelV20Agent(**RULES)


def fixture(name):
    path = FIXTURES / (name + '.json')
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())['fixtures'][path.name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']
    data = json.loads(path.read_text())
    assert hashlib.sha256(data['public_wire'].encode()).hexdigest() == manifest['wire_sha256']
    assert data['rules'] == RULES
    obs = read_observation(io.StringIO(data['public_wire']), *data['shape'])
    return data, obs, jnp.asarray(data['action_key'], jnp.uint32)


def exact_tuple(result, data):
    action, memory, telemetry = result
    np.testing.assert_array_equal(action, data['expected_parent_action'])
    same_tree(memory, restore(memory, data['expected_parent_memory']))
    assert telemetry.keys() == data['expected_parent_telemetry'].keys()
    for key, value in data['expected_parent_telemetry'].items():
        np.testing.assert_array_equal(telemetry[key], value, err_msg=key)


def assert_inactive(result, parent):
    action, memory, telemetry = result
    same_tree((action, memory.parent), parent[:2])
    assert not telemetry['branch_action_issued']
    assert not telemetry['branch_override']
    assert telemetry['branch_parent_action_issued']
    assert memory.phase == 0 and memory.budget == 0
    assert memory.objective == memory.rally == memory.packet == -1
    assert len(jax.tree.leaves(memory)) == 28
    assert all(np.asarray(x).shape == () and np.asarray(x).dtype == np.int32 for x in jax.tree.leaves(memory))
    inherited = {k: v for k, v in telemetry.items() if not k.startswith('branch_')}
    same_tree(inherited, parent[2])


@pytest.mark.parametrize('name', V10_NAMES)
def test_disabled_exact_original_v10_full_tuple(name):
    data, obs, key = fixture(name)
    memory = restore(DISABLED.initial_memory(data['shape']), data['incoming_memory'])
    result = DISABLED.step(obs, key, memory)
    exact_tuple(result, data)
    assert len(jax.tree.leaves(result[1])) == 19


@pytest.mark.parametrize('name', V10_NAMES)
def test_enabled_preserves_original_parent_on_saved_inactive_frames(name):
    data, obs, key = fixture(name)
    native = restore(PARENT.initial_memory(data['shape']), data['incoming_memory'])
    memory = ENABLED.initial_memory(data['shape'])._replace(parent=native, last_turn=jnp.int32(data['turn'] - 1))
    result = ENABLED.step(obs, key, memory)
    # This is an archived oracle, not a second newly computed parent call.
    archived_parent = (jnp.asarray(data['expected_parent_action'], jnp.int32),
                       restore(native, data['expected_parent_memory']),
                       {k: jnp.asarray(v) for k, v in data['expected_parent_telemetry'].items()})
    assert_inactive(result, archived_parent)
    if name in ('v10-amin-348', 'v10-amin-352', 'v10-amin-359', 'v10-juraj-420'):
        assert result[2]['branch_priority']
        assert result[2]['intercept_home_deficit'] > 0


@pytest.mark.parametrize('name', LEGACY_NAMES)
def test_archived_v10_v6_is_checked_against_its_actual_parent(name):
    data, obs, key = fixture(name)
    assert data['parent_variant'] == 'v10-v6'
    native = restore(LEGACY.initial_memory(data['shape']), data['incoming_memory'])
    assert len(jax.tree.leaves(native)) == 7
    exact_tuple(LEGACY.step(obs, key, native), data)


@pytest.mark.parametrize('name', LEGACY_NAMES)
def test_fresh_default_parent_on_legacy_public_board_is_not_archived_oracle(name):
    data, obs, key = fixture(name)
    # Explicit new unit-call context: empty V10 strategic/offense state, with the
    # true legacy defender inserted. This native19 state was NOT recorded.
    native = PARENT.initial_memory(data['shape'])
    defender = restore(native.base.defense, data['incoming_memory'])
    native = native._replace(base=native.base._replace(defense=defender, last_turn=jnp.int32(data['turn'] - 1)),
                             last_turn=jnp.int32(data['turn'] - 1),
                             height=jnp.int32(data['shape'][0]), width=jnp.int32(data['shape'][1]))
    memory = ENABLED.initial_memory(data['shape'])._replace(parent=native, last_turn=jnp.int32(data['turn'] - 1))
    fresh_parent = PARENT.step(obs, key, native)
    result = ENABLED.step(obs, key, memory)
    assert_inactive(result, fresh_parent)
    if data['turn'] == 477:
        assert defender.defender >= 0 and result[2]['branch_priority']
    if data['turn'] == 805:
        assert result[2]['intercept_home_deficit'] > 0 and result[2]['branch_priority']

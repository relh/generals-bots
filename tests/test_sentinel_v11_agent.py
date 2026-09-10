"""Capture arbitration is an executable hypothesis with exact parent controls."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v11_agent import SentinelV11Agent
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore


def recorded(name, turn=129):
    data = json.loads((Path(__file__).parent / "fixtures" / f"capture-priority-{name}-{turn}.json").read_text())
    obs = read_observation(io.StringIO(data["wire"]), *data["shape"])
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800)
    memory = restore(agent.initial_memory(obs.armies.shape), data["memory_before"])
    return data, obs, jnp.array(data["action_key"], jnp.uint32), memory


def test_recorded_juraj_capture_changes_only_the_unissued_offensive_plan():
    _, obs, key, memory = recorded("juraj")
    before = jax.tree.map(lambda x: None if x is None else np.array(x), obs)
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    old_action, old_memory, _ = parent.step(obs, key, memory)
    action, returned, tel = agent.step(obs, key, memory)
    np.testing.assert_array_equal(old_action, [0, 15, 10, 3, 0])
    np.testing.assert_array_equal(action, [0, 8, 8, 0, 0])
    assert tel["capture_priority_issued"] and not tel["capture_parent_action_issued"]
    assert tel["capture_planned_eta"] == 8
    assert returned.base.phase == 0 and returned.base.packet == -1
    assert returned.base.last_turn == 129
    same_tree(returned.base.defense, old_memory.base.defense)
    for name in ("enemy_general", "last_turn", "height", "width"):
        np.testing.assert_array_equal(getattr(returned, name), getattr(old_memory, name))
    same_tree(obs, before)


@pytest.mark.parametrize("name", ["juraj", "amin"])
def test_disabled_is_exact_frozen_v10_including_telemetry(name):
    _, obs, key, memory = recorded(name)
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800, preserve_enemy_captures=False)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    same_tree(agent.step(obs, key, memory), parent.step(obs, key, memory))


def test_known_productive_amin_plan_can_defer_a_neutral_capture():
    _, obs, key, memory = recorded("amin")
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    action, returned, tel = agent.step(obs, key, memory)
    expected, old_memory, _ = parent.step(obs, key, memory)
    np.testing.assert_array_equal(action, [0, 15, 13, 2, 0])
    same_tree((action, returned), (expected, old_memory))
    assert tel["offense_started"] and not tel["capture_priority_issued"]
    assert not tel["capture_immediate_available"]


def test_existing_plan_continues_despite_an_immediate_enemy_capture():
    _, obs, key, memory = recorded("juraj", 130)
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    action, returned, tel = agent.step(obs, key, memory)
    old_action, old_memory, _ = parent.step(obs, key, memory)
    np.testing.assert_array_equal(action, [0, 15, 11, 3, 0])
    same_tree((action, returned), (old_action, old_memory))
    assert tel["offense_continued"] and tel["capture_immediate_available"]
    assert not tel["capture_priority_issued"] and returned.base.phase > 0


def test_batched_structure_target_preserves_parent_and_scalar_memory_schema():
    data, obs, key, memory = recorded("juraj")
    target = data["expected_target"]["position"]
    # A strategic structure is not interchangeable with an ordinary tile.
    structured = obs._replace(castles=obs.castles.at[tuple(target)].set(True))
    agent = SentinelV11Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), obs, structured)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    actions, returned, tel = jax.jit(jax.vmap(agent.step))(batch, jnp.stack((key, key)), memories)
    np.testing.assert_array_equal(tel["capture_priority_issued"], [True, False])
    expected, old_memory, _ = parent.step(structured, key, memory)
    same_tree((actions[1], jax.tree.map(lambda x: x[1], returned)), (expected, old_memory))
    assert len(jax.tree.leaves(returned)) == 19
    assert all(x.shape == (2,) and x.dtype == jnp.int32 for x in jax.tree.leaves(returned))

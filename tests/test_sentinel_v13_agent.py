"""Collection-only arbitration preserves ready deployment, not all useful plans."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v13_agent import SentinelV13Agent
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore


def recorded(name, turn=129):
    data = json.loads((Path(__file__).parent / "fixtures" / f"capture-priority-{name}-{turn}.json").read_text())
    obs = read_observation(io.StringIO(data["wire"]), *data["shape"])
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800)
    memory = restore(agent.initial_memory(obs.armies.shape), data["memory_before"])
    return data, obs, jnp.array(data["action_key"], jnp.uint32), memory


def test_recorded_juraj_capture_changes_only_the_unissued_offensive_plan():
    _, obs, key, memory = recorded("juraj")
    before = jax.tree.map(lambda x: None if x is None else np.array(x), obs)
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    old_action, old_memory, _ = parent.step(obs, key, memory)
    action, returned, tel = agent.step(obs, key, memory)
    np.testing.assert_array_equal(old_action, [0, 15, 10, 3, 0])
    np.testing.assert_array_equal(action, [0, 8, 8, 0, 0])
    assert tel["collection_capture_priority_issued"] and not tel["collection_capture_parent_action_issued"]
    assert tel["collection_capture_planned_eta"] == 8
    assert returned.base.phase == 0 and returned.base.packet == -1
    assert returned.base.last_turn == 129
    same_tree(returned.base.defense, old_memory.base.defense)
    for name in ("enemy_general", "last_turn", "height", "width"):
        np.testing.assert_array_equal(getattr(returned, name), getattr(old_memory, name))
    same_tree(obs, before)


@pytest.mark.parametrize("name", ["juraj", "amin"])
def test_disabled_is_exact_frozen_v10_including_telemetry(name):
    _, obs, key, memory = recorded(name)
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800, preserve_collection_captures=False)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    same_tree(agent.step(obs, key, memory), parent.step(obs, key, memory))


def test_known_productive_amin_plan_can_defer_a_neutral_capture():
    _, obs, key, memory = recorded("amin")
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    action, returned, tel = agent.step(obs, key, memory)
    expected, old_memory, _ = parent.step(obs, key, memory)
    np.testing.assert_array_equal(action, [0, 15, 13, 2, 0])
    same_tree((action, returned), (expected, old_memory))
    assert tel["offense_started"] and not tel["collection_capture_priority_issued"]
    assert not tel["collection_capture_immediate_available"]


def test_existing_plan_continues_despite_an_immediate_enemy_capture():
    _, obs, key, memory = recorded("juraj", 130)
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    action, returned, tel = agent.step(obs, key, memory)
    old_action, old_memory, _ = parent.step(obs, key, memory)
    np.testing.assert_array_equal(action, [0, 15, 11, 3, 0])
    same_tree((action, returned), (old_action, old_memory))
    assert tel["offense_continued"] and tel["collection_capture_immediate_available"]
    assert not tel["collection_capture_priority_issued"] and returned.base.phase > 0


def test_batched_structure_target_preserves_parent_and_scalar_memory_schema():
    data, obs, key, memory = recorded("juraj")
    target = data["expected_target"]["position"]
    # A strategic structure is not interchangeable with an ordinary tile.
    structured = obs._replace(castles=obs.castles.at[tuple(target)].set(True))
    agent = SentinelV13Agent(build_castles=True, deathtouch_turn=800)
    parent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    batch = jax.tree.map(lambda a, b: jnp.stack((a, b)), obs, structured)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    actions, returned, tel = jax.jit(jax.vmap(agent.step))(batch, jnp.stack((key, key)), memories)
    np.testing.assert_array_equal(tel["collection_capture_priority_issued"], [True, False])
    expected, old_memory, _ = parent.step(structured, key, memory)
    same_tree((actions[1], jax.tree.map(lambda x: x[1], returned)), (expected, old_memory))
    assert len(jax.tree.leaves(returned)) == 19
    assert all(x.shape == (2,) and x.dtype == jnp.int32 for x in jax.tree.leaves(returned))


@pytest.mark.parametrize("turn", [163, 193])
def test_actual_ready_deployment_and_productive_collection_counterexample(turn):
    data = json.loads((Path(__file__).parent / "fixtures" / f"collection-priority-amin-{turn}.json").read_text())
    obs = read_observation(io.StringIO(data["public_wire"]), *data["shape"])
    agent = SentinelV13Agent(**data["rules"])
    memory = restore(agent.initial_memory(obs.armies.shape), data["incoming_memory"])
    action, returned, tel = agent.step(obs, jnp.array(data["action_key"], jnp.uint32), memory)
    parent_action = jnp.stack([tel["collection_parent_" + name] for name in
                               ("kind", "row", "column", "direction", "split")])
    np.testing.assert_array_equal(parent_action, data["expected_parent_action"])
    expected_memory = restore(agent.initial_memory(obs.armies.shape), data["original_returned_memory"])
    if turn == 163:
        # V11 diverted this ready army. V13 must issue the recorded direct plan.
        assert tel["offense_direct_started"] and not tel["offense_collecting"]
        assert tel["collection_capture_immediate_available"]
        assert not tel["collection_capture_priority_issued"]
        same_tree((action, returned), (parent_action, expected_memory))
    else:
        # This collection really succeeded in the control. Rejecting it is a
        # deliberate strategic risk, not proof that the collection was useless.
        assert tel["offense_collecting"] and tel["collection_capture_priority_issued"]
        reference = jnp.stack([tel["capture_reference_" + name] for name in
                               ("kind", "row", "column", "direction", "split")])
        np.testing.assert_array_equal(reference, data["same_observation_v6_proposal"]["action"])
        np.testing.assert_array_equal(action, reference)
        assert returned.base.phase == 0 and returned.base.packet == -1
        same_tree(returned.base.defense, expected_memory.base.defense)
        assert not tel["collection_capture_parent_action_issued"]

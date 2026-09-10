"""Real public states check action budgets and actual transported memory."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v14_agent import SentinelV14Agent, ServiceMemory
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
AGENT = SentinelV14Agent(**RULES)
PARENT = SentinelV10Agent(**RULES)
DISABLED = SentinelV14Agent(**RULES, balance_collection=False)


def recorded(name):
    data = json.loads((Path(__file__).parent / "fixtures" / "collection-service" / f"{name}.json").read_text())
    obs = read_observation(io.StringIO(data["public_wire"]), *data["shape"])
    if name == "juraj106":
        memory = ServiceMemory(restore(PARENT.initial_memory(obs.armies.shape), data["incoming_memory"]),
                               jnp.int32(data["diagnostic_debt"]))
        expected_memory = data["expected_parent_returned_memory"]
    else:
        memory = restore(AGENT.initial_memory(obs.armies.shape), data["incoming_memory"])
        expected_memory = data["original_parent_returned_memory"]
    expected_memory = restore(PARENT.initial_memory(obs.armies.shape), expected_memory)
    return data, obs, jnp.array(data["action_key"], jnp.uint32), memory, expected_memory


def parent_vector(tel):
    return jnp.stack([tel["service_parent_" + name] for name in ("kind", "row", "column", "direction", "split")])


@pytest.mark.parametrize("name", ["juraj106", "amin193"])
def test_actual_new_collection_defers_for_campaign_and_reconciles_memory(name):
    data, obs, key, memory, expected_memory = recorded(name)
    before = jax.tree.map(lambda x: None if x is None else np.array(x), obs)
    action, returned, tel = AGENT.step(obs, key, memory)
    np.testing.assert_array_equal(parent_vector(tel), data["expected_parent_action"])
    expected = data["expected_reference_action"] if name == "juraj106" else data["same_observation_v6_proposal"]
    np.testing.assert_array_equal(action, expected)
    assert tel["service_collection_deferred"] and not tel["service_parent_action_issued"]
    assert tel["service_campaign_served"] and not tel["service_collection_charged"]
    assert returned.collection_debt == 2 and tel["service_debt_repaid"] == 1
    assert returned.base.base.phase == 0 and returned.base.base.packet == -1
    same_tree(returned.base.base.defense, expected_memory.base.defense)
    for field in ("enemy_general", "last_turn", "height", "width"):
        np.testing.assert_array_equal(getattr(returned.base, field), getattr(expected_memory, field))
    same_tree(obs, before)


@pytest.mark.parametrize("name,debt", [("amin129", 1), ("amin130", 2), ("amin284", 14)])
def test_start_continuation_and_same_action_collection_are_charged_once(name, debt):
    data, obs, key, memory, expected_memory = recorded(name)
    action, returned, tel = AGENT.step(obs, key, memory)
    same_tree((action, returned.base), (jnp.array(data["expected_parent_action"]), expected_memory))
    assert tel["service_collection_charged"] and not tel["service_campaign_served"]
    assert not tel["service_collection_deferred"] and tel["service_parent_action_issued"]
    assert returned.collection_debt == debt and tel["service_debt_repaid"] == 0
    if name == "amin130":
        assert tel["offense_continued"] and memory.base.base.phase > 0
    if name == "amin284":
        np.testing.assert_array_equal(action, data["same_observation_v6_proposal"])
        assert tel["offense_started"] and returned.base.base.phase > 0


@pytest.mark.parametrize("name", ["juraj106", "amin193"])
def test_disabled_is_exact_v10_with_original_nineteen_field_memory(name):
    _, obs, key, memory, _ = recorded(name)
    same_tree(DISABLED.initial_memory(obs.armies.shape), PARENT.initial_memory(obs.armies.shape))
    same_tree(DISABLED.step(obs, key, memory.base), PARENT.step(obs, key, memory.base))


@pytest.mark.parametrize("reset", ["time", "gap", "shape"])
def test_incoming_reset_discards_old_service_debt(reset):
    _, obs, key, memory, _ = recorded("amin193")
    if reset == "time":
        base = memory.base._replace(last_turn=obs.timestep)
    elif reset == "gap":
        base = memory.base._replace(last_turn=obs.timestep - 3)
    else:
        base = memory.base._replace(width=memory.base.width + 1)
    action, returned, tel = AGENT.step(obs, key, ServiceMemory(base, jnp.int32(99)))
    expected, parent_memory, _ = PARENT.step(obs, key, base)
    assert tel["service_debt_reset"] and tel["service_debt_before"] == 0
    assert not tel["service_collection_deferred"]
    same_tree((action, returned.base), (expected, parent_memory))
    assert returned.collection_debt == int(tel["service_collection_charged"])


def test_batched_budget_and_reset_are_isolated_and_keep_twenty_typed_fields():
    _, obs, key, memory, _ = recorded("amin193")
    reset = ServiceMemory(memory.base._replace(last_turn=obs.timestep), memory.collection_debt)
    batch = jax.tree.map(lambda x: jnp.stack((x, x)), obs)
    memories = jax.tree.map(lambda x, y: jnp.stack((x, y)), memory, reset)
    _, returned, tel = jax.jit(jax.vmap(AGENT.step))(batch, jnp.stack((key, key)), memories)
    np.testing.assert_array_equal(tel["service_collection_deferred"], [True, False])
    np.testing.assert_array_equal(returned.collection_debt, [2, 1])
    leaves = jax.tree.leaves(returned)
    assert len(leaves) == 20 and all(x.shape == (2,) and x.dtype == jnp.int32 for x in leaves)

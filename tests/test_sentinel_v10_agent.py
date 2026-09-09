"""Home mobilization uses real public force and reconciles unissued plans."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v6_agent import SentinelV6Agent
from generals.agents.sentinel_v9_agent import SentinelV9Agent
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree


def restore(template, data):
    if hasattr(template, "_fields"):
        return type(template)(*(restore(getattr(template, k), data[k]) for k in template._fields))
    return jnp.int32(data)


@pytest.fixture(scope="module")
def fixture():
    data = json.loads((Path(__file__).parent / "fixtures/mobilization-public-1027.json").read_text())
    obs = read_observation(io.StringIO(data["wire"]), 18, 21)
    key = jnp.array(data["action_key"], dtype=jnp.uint32)
    return data, obs, key


@pytest.mark.parametrize("parent_version", [6, 9])
def test_recorded_home_route_is_faster_and_clears_unissued_field_plan(fixture, parent_version):
    data, obs, key = fixture
    agent = SentinelV10Agent(parent_version=parent_version, build_castles=True, deathtouch_turn=800)
    before = data["memory_before"] if parent_version == 9 else data["memory_before"]["base"]["defense"]
    memory = restore(agent.initial_memory(obs.armies.shape), before)
    old_obs = jax.tree.map(lambda x: None if x is None else np.array(x), obs)
    action, returned, tel = agent.step(obs, key, memory)
    np.testing.assert_array_equal(action, [0, 14, 5, 0, 0])
    assert tel["mobilization_issued"] and not tel["mobilization_parent_action_issued"]
    assert tel["mobilization_home_eta"] == 3
    assert tel["mobilization_arrival_army"] == 60
    assert tel["mobilization_donation"] == 58
    assert tel["mobilization_home_coverage_safe"]
    defense = returned if parent_version == 6 else returned.base.defense
    assert defense.defender == -1 and defense.target == -1 and defense.last_turn == 1027
    if parent_version == 9:
        assert returned.base.packet == -1 and returned.base.last_turn == 1027
        assert returned.enemy_general == -1 and returned.last_turn == 1027
        assert returned.height == 18 and returned.width == 21
    same_tree(obs, old_obs)
    assert all(np.asarray(x).shape == () and np.asarray(x).dtype == np.int32 for x in jax.tree.leaves(returned))


@pytest.mark.parametrize("parent_version", [6, 9])
def test_disabled_preserves_independent_parent_exactly(fixture, parent_version):
    data, obs, key = fixture
    agent = SentinelV10Agent(
        parent_version=parent_version, build_castles=True, deathtouch_turn=800, mobilize_home=False
    )
    cls = SentinelV6Agent if parent_version == 6 else SentinelV9Agent
    parent = cls(build_castles=True, deathtouch_turn=800)
    before = data["memory_before"] if parent_version == 9 else data["memory_before"]["base"]["defense"]
    memory = restore(agent.initial_memory(obs.armies.shape), before)
    same_tree(agent.step(obs, key, memory), parent.step(obs, key, memory))


@pytest.mark.parametrize("condition", ["before_deathtouch", "no_deathtouch", "no_home_army", "active_defender"])
def test_noneligible_conditions_preserve_parent_action_memory(fixture, condition):
    data, obs, key = fixture
    touch = None if condition == "no_deathtouch" else 1100 if condition == "before_deathtouch" else 800
    agent = SentinelV10Agent(build_castles=True, deathtouch_turn=touch)
    memory = restore(agent.initial_memory(obs.armies.shape), data["memory_before"])
    if condition == "no_home_army":
        obs = obs._replace(armies=obs.armies.at[14, 5].set(1))
    if condition == "active_defender":
        defense = memory.base.defense._replace(defender=jnp.int32(219))
        memory = memory._replace(base=memory.base._replace(defense=defense))
    action, returned, tel = agent.step(obs, key, memory)
    expected, old_memory, old_tel = agent.base.step(obs, key, memory)
    same_tree((action, returned), (expected, old_memory))
    assert not tel["mobilization_issued"] and not tel["mobilization_eligible"]
    for name, value in old_tel.items():
        np.testing.assert_array_equal(tel[name], value)


def test_invalid_parent_rejected():
    with pytest.raises(ValueError, match="parent_version"):
        SentinelV10Agent(parent_version=8)


def test_winning_capture_and_build_are_parent_decisions():
    agent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    key = jax.random.PRNGKey(0)
    observations = [
        board([(5, 0, 20, True), (3, 4, 80, False)], shape=(6, 6), time=800),
        board([(5, 0, 20, True), (3, 4, 2, False)], [(2, 4, 100, True)], shape=(6, 6), time=800),
    ]
    for i, obs in enumerate(observations):
        memory = agent.initial_memory(obs.armies.shape)
        action, new, tel = agent.step(obs, key, memory)
        old_action, old_memory, _ = agent.base.step(obs, key, memory)
        same_tree((action, new), (old_action, old_memory))
        assert not tel["mobilization_issued"]
        if i == 0:
            assert action[0] == 2
        else:
            np.testing.assert_array_equal(action, [0, 3, 4, 0, 0])


def test_batched_active_and_inactive_home_force(fixture):
    data, obs, key = fixture
    agent = SentinelV10Agent(build_castles=True, deathtouch_turn=800)
    memory = restore(agent.initial_memory(obs.armies.shape), data["memory_before"])
    depleted = obs._replace(armies=obs.armies.at[14, 5].set(1))
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), obs, depleted)
    memories = jax.tree.map(lambda x: jnp.stack([x, x]), memory)
    actions, returned, tel = jax.jit(jax.vmap(agent.step))(batch, jnp.stack([key, key]), memories)
    np.testing.assert_array_equal(actions[0], [0, 14, 5, 0, 0])
    np.testing.assert_array_equal(tel["mobilization_issued"], [True, False])
    assert len(jax.tree.leaves(returned)) == 19
    assert all(x.shape == (2,) and x.dtype == jnp.int32 for x in jax.tree.leaves(returned))

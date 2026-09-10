"""Direct expansion, preserved parent priorities and exact production timing."""

import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v15_agent import FrontierMemory, SentinelV15Agent, _frontier_proposal
from generals.core import game
from tests.test_multiplayer import board as state_board
from tests.test_multiplayer import give, move, passes
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v10_agent import restore

RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
KEY = jax.random.PRNGKey(42)


def fixture(name="juraj17"):
    data = json.loads((Path(__file__).parent / f"fixtures/frontier/{name}.json").read_text())
    obs = read_observation(io.StringIO(data["public_wire"]), *data["shape"])
    parent = SentinelV10Agent(**RULES)
    memory = restore(parent.initial_memory(obs.armies.shape), data["incoming_memory"])
    return data, obs, memory


def test_recorded_opening_expands_instead_of_transport_and_preserves_parent_state():
    data, obs, memory = fixture()
    agent = SentinelV15Agent(**RULES, frontier_mode="fanout")
    action, returned, tel = agent.step(
        obs, jnp.array(data["action_key"], jnp.uint32), FrontierMemory(memory, jnp.int32(0)),
    )
    np.testing.assert_array_equal(action, [0, 9, 17, 3, 1])
    np.testing.assert_array_equal(
        [tel["frontier_parent_" + k] for k in ("kind", "row", "column", "direction", "split")],
        data["expected_parent_action"],
    )
    assert tel["frontier_issued"] and not tel["frontier_parent_action_issued"]
    assert tel["frontier_next_land_tick"] == 50 and tel["frontier_steps_to_land_tick"] == 33
    same_tree(returned.base, restore(memory, data["expected_parent_memory"]))


def test_disabled_preserves_complete_frozen_tuple():
    data, obs, memory = fixture()
    key = jnp.array(data["action_key"], jnp.uint32)
    agent = SentinelV15Agent(**RULES, expand_frontier=False)
    same_tree(agent.step(obs, key, memory), SentinelV10Agent(**RULES).step(obs, key, memory))


@pytest.mark.parametrize("name,issued", [("juraj17", False), ("amin49", True), ("amin129", False), ("amin142", False)])
def test_tick_mode_uses_income_boundary_and_preserves_productive_contact_plans(name, issued):
    data, obs, base = fixture(name)
    agent = SentinelV15Agent(**RULES)
    action, returned, tel = agent.step(obs, jnp.array(data["action_key"], jnp.uint32),
                                     FrontierMemory(base, jnp.int32(0)))
    assert bool(tel["frontier_issued"]) == issued
    if issued:
        np.testing.assert_array_equal(action, [0, 8, 17, 0, 1])
        assert obs.armies[8, 17] - obs.armies[8, 17] // 2 >= tel["general_reserve"]
        assert tel["frontier_steps_to_land_tick"] == 1
    else:
        same_tree(action, jnp.array(data["expected_parent_action"]))
        same_tree(returned.base, restore(base, data["expected_parent_memory"]))
        if name != "juraj17":
            assert tel["frontier_contact_seen"] and tel["offense_started"]


def test_remembered_contact_survives_fog_and_gaps_but_resets_with_new_game():
    data, obs, base = fixture("amin49")
    agent = SentinelV15Agent(**RULES)
    key = jnp.array(data["action_key"], jnp.uint32)
    for prior in (base, base._replace(last_turn=jnp.int32(40))):
        _, returned, tel = agent.step(obs, key, FrontierMemory(prior, jnp.int32(1)))
        assert returned.contact_seen == 1 and not tel["frontier_issued"]
    for prior in (base._replace(last_turn=obs.timestep), base._replace(width=base.width + 1)):
        _, returned, tel = agent.step(obs, key, FrontierMemory(prior, jnp.int32(1)))
        assert returned.contact_seen == 0 and tel["frontier_contact_reset"]
        assert tel["frontier_issued"]


def test_contact_modes_batch_independently_with_twenty_int32_fields():
    data, obs, base = fixture("amin49")
    agent = SentinelV15Agent(**RULES)
    observations = jax.tree.map(lambda x: jnp.stack((x, x)), obs)
    memory = FrontierMemory(jax.tree.map(lambda x: jnp.stack((x, x)), base), jnp.array([0, 1], jnp.int32))
    key = jnp.array(data["action_key"], jnp.uint32)
    _, returned, tel = jax.jit(jax.vmap(agent.step))(observations, jnp.stack((key, key)), memory)
    np.testing.assert_array_equal(tel["frontier_issued"], [True, False])
    leaves = jax.tree.leaves(returned)
    assert len(leaves) == 20 and all(x.dtype == jnp.int32 and x.shape == (2,) for x in leaves)


def test_half_general_expansion_keeps_actual_reserve():
    obs = board([(0, 0, 6, True)], shape=(4, 4))
    action, available, _, _ = _frontier_proposal(obs, 3)
    assert available and action[4] == 1
    assert not _frontier_proposal(obs._replace(armies=obs.armies.at[0, 0].set(4)), 3)[1]


@pytest.mark.parametrize("obstacle", ["fog", "castle", "mountain", "hidden_structure", "defended"])
def test_empty_looking_or_contested_destinations_are_not_free_expansion(obstacle):
    # Only the eastern destination is accessible from the sole eligible donor.
    obs = board([(0, 0, 10, True), (3, 3, 3, False)], shape=(7, 7),
                mountains=[(2, 3), (4, 3), (3, 2)])
    if obstacle == "fog":
        obs = obs._replace(fog_cells=obs.fog_cells.at[3, 4].set(True))
    elif obstacle == "castle":
        obs = obs._replace(castles=obs.castles.at[3, 4].set(True))
    elif obstacle == "mountain":
        obs = obs._replace(mountains=obs.mountains.at[3, 4].set(True))
    elif obstacle == "hidden_structure":
        obs = obs._replace(structures_in_fog=obs.structures_in_fog.at[3, 4].set(True))
    elif obstacle == "defended":
        obs = obs._replace(armies=obs.armies.at[3, 4].set(1))
    assert not _frontier_proposal(obs)[1]


def test_locality_excludes_structures_and_large_packets():
    obs = board([(0, 0, 100, True), (1, 1, 20, False), (3, 3, 3, False)],
                castles=[(2, 2, 3)], shape=(7, 7))
    obs = obs._replace(owned_cells=obs.owned_cells.at[2, 2].set(True),
                       neutral_cells=obs.neutral_cells.at[2, 2].set(False))
    action, available, _, _ = _frontier_proposal(obs)
    assert available
    np.testing.assert_array_equal(action[1:3], [3, 3])
    assert action[3] in (0, 2) and action[4] == 1
    assert not _frontier_proposal(obs._replace(generals=jnp.zeros_like(obs.generals)))[1]


@pytest.mark.parametrize("turn,expected_destination", [(49, 2), (50, 1)])
def test_expansion_before_land_tick_earns_income_in_ffa(turn, expected_destination):
    state = state_board({0: (0, 0), 1: (0, 6), 2: (6, 6)}, size=7)
    state = give(state, 0, (3, 3), 2)._replace(time=jnp.int32(turn))
    actions = passes(3).at[0].set(move(3, 3, 3))
    after, _ = game.step(state, actions)
    assert after.ownership[0, 3, 4]
    assert after.armies[3, 4] == expected_destination
    assert after.armies[3, 3] == expected_destination


def test_general_first_production_is_step_two_not_first_land_tick():
    state = state_board({0: (0, 0), 1: (3, 3)}, size=4)
    one, _ = game.step(state, passes(2))
    two, _ = game.step(one, passes(2))
    assert one.armies[0, 0] == 1 and two.armies[0, 0] == 2

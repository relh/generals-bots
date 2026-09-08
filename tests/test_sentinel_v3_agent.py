"""State/history contracts and defensive tactics, independent of win-rate gates."""

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_agent import SentinelAgent
from generals.agents.sentinel_v3_agent import SentinelV3Agent
from tests.test_sentinel_agent import board

KEY = jax.random.PRNGKey(0)


def step(agent, obs, memory=None):
    if memory is None:
        memory = agent.initial_memory(obs.armies.shape)
    return agent.step(obs, KEY, memory)


def test_disabled_ablations_match_frozen_policy():
    agent = SentinelV3Agent(remember_threats=False, sustained_defense=False, build_castles=True)
    baseline = SentinelAgent(build_castles=True)
    observations = [
        board([(0, 0, 10, True), (2, 2, 80, False)]),
        board([(1, 1, 23, True), (0, 1, 2, False)], [(2, 2, 13, False)]),
        board([(1, 1, 3, True), (0, 2, 15, False)], [(1, 2, 10, False)]),
    ]
    for obs in observations:
        np.testing.assert_array_equal(step(agent, obs)[0], baseline.act(obs, KEY))


def test_retains_rescued_garrison_against_two_step_wave():
    obs = board([(1, 1, 23, True), (0, 1, 2, False)], [(2, 2, 13, False)])
    action, _, telemetry = step(SentinelV3Agent(), obs)
    assert telemetry["general_reserve"] >= 16
    assert telemetry["defense_override"]
    assert action[0] == 1 or tuple(np.asarray(action[1:3])) != (1, 1)


def test_recalls_friendly_stack_before_distant_visible_invasion_arrives():
    obs = board(
        [(3, 0, 20, True), (3, 1, 1, False), (3, 2, 1, False), (3, 3, 25, False)], [(0, 6, 70, False)], shape=(7, 7)
    )
    action, _, telemetry = step(SentinelV3Agent(), obs)
    np.testing.assert_array_equal(action, [0, 3, 3, 2, 0])
    assert telemetry["visible_reserve"] > 50


def test_memory_survives_fog_but_visible_empty_cells_contradict_it():
    agent = SentinelV3Agent(sustained_defense=False)
    seen = board([(3, 0, 20, True), (3, 1, 1, False)], [(1, 4, 50, False)], shape=(7, 7))
    _, memory, _ = step(agent, seen)
    empty = board([(3, 0, 20, True), (3, 1, 1, False)], shape=(7, 7), time=1)
    fog = jnp.zeros((7, 7), bool).at[0:3, 3:6].set(True)
    hidden = empty._replace(fog_cells=fog, neutral_cells=empty.neutral_cells & ~fog, opponent_army_count=jnp.int32(50))
    _, memory, telemetry = step(agent, hidden, memory)
    assert memory.threat_army[1, 4] > 0
    assert telemetry["remembered_reserve"] > 0
    # Same cell now observed empty; other possible hidden locations may remain.
    cleared = hidden._replace(
        timestep=jnp.int32(2), fog_cells=fog.at[1, 4].set(False), neutral_cells=hidden.neutral_cells.at[1, 4].set(True)
    )
    _, memory, _ = step(agent, cleared, memory)
    assert memory.threat_army[1, 4] == 0
    assert jnp.max(memory.threat_army) > 0
    fully_seen = empty._replace(timestep=jnp.int32(3), opponent_army_count=jnp.int32(50))
    _, memory, telemetry = step(agent, fully_seen, memory)
    assert jnp.max(memory.threat_army) == 0
    assert telemetry["remembered_reserve"] == 0


def test_old_threat_expires_and_does_not_sum_ambiguous_copies():
    agent = SentinelV3Agent(sustained_defense=False)
    obs = board([(3, 0, 10, True), (3, 1, 1, False)], [(1, 4, 50, False)], shape=(7, 7))
    _, memory, _ = step(agent, obs)
    empty = board([(3, 0, 10, True), (3, 1, 1, False)], shape=(7, 7), time=25)
    fog = ~empty.owned_cells
    empty = empty._replace(fog_cells=fog, neutral_cells=jnp.zeros((7, 7), bool), opponent_army_count=jnp.int32(50))
    _, memory, telemetry = step(agent, empty, memory)
    assert jnp.max(memory.threat_army) == 0
    assert telemetry["general_reserve"] == 3


def test_sustained_defense_holds_then_releases_reserve():
    agent = SentinelV3Agent(remember_threats=False)
    seen = board([(1, 1, 23, True), (0, 1, 2, False)], [(2, 2, 13, False)])
    _, memory, initial = step(agent, seen)
    clear = board([(1, 1, 23, True), (0, 1, 2, False)], time=1)
    _, memory, held = step(agent, clear, memory)
    assert held["general_reserve"] == initial["general_reserve"]
    assert held["remembered_reserve"] == 0
    for turn in range(2, 30):
        _, memory, telemetry = step(agent, clear._replace(timestep=jnp.int32(turn)), memory)
    assert telemetry["general_reserve"] == 3
    assert not telemetry["defense_active"]


def test_preserves_founding_corridor_and_immediate_interception():
    agent = SentinelV3Agent()
    founding = board([(1, 1, 2, True)], [(2, 1, 2, True)])
    assert step(agent, founding)[0][0] == 0
    corridor = board([(3, 1, 8, True)], [(2, 2, 8, True)], mountains=[(3, 0), (3, 2)])
    np.testing.assert_array_equal(step(agent, corridor)[0], [0, 3, 1, 0, 1])
    rescue = board([(1, 1, 3, True), (0, 2, 15, False)], [(1, 2, 10, False)], time=800)
    np.testing.assert_array_equal(step(SentinelV3Agent(deathtouch_turn=800), rescue)[0], [0, 0, 2, 1, 0])


def test_winning_capture_overrides_distant_defense_and_build_rules_hold():
    obs = board([(0, 0, 5, True), (2, 2, 20, False)], [(2, 3, 10, True), (3, 0, 40, False)])
    np.testing.assert_array_equal(step(SentinelV3Agent(), obs)[0], [0, 2, 2, 3, 0])
    build = board([(0, 0, 20, True), (2, 2, 80, False)])
    assert step(SentinelV3Agent(build_castles=True), build)[0][0] == 2
    assert step(SentinelV3Agent(build_castles=False), build)[0][0] != 2
    late = build._replace(timestep=jnp.int32(1150))
    assert step(SentinelV3Agent(build_castles=True), late)[0][0] != 2


def test_jit_vmap_memories_are_independent():
    agent = SentinelV3Agent()
    threat = board([(1, 1, 23, True), (0, 1, 2, False)], [(2, 2, 13, False)])
    clear = board([(1, 1, 23, True), (0, 1, 2, False)])
    observations = jax.tree.map(lambda x, y: jnp.stack((x, y)), threat, clear)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), agent.initial_memory((4, 4)))
    actions, memories, telemetry = jax.jit(jax.vmap(agent.step))(observations, jax.random.split(KEY, 2), memories)
    assert actions.shape == (2, 5)
    assert memories.threat_army.shape == (2, 4, 4)
    assert telemetry["general_reserve"][0] > telemetry["general_reserve"][1]


def test_planning_buffer_alone_does_not_recall_forward_army():
    obs = board(
        [(3, 0, 10, True), (3, 1, 1, False), (3, 2, 1, False), (3, 3, 12, False)], [(0, 9, 26, False)], shape=(10, 10)
    )
    action, _, telemetry = step(SentinelV3Agent(), obs)
    assert telemetry["visible_reserve"] == 14
    assert not telemetry["defense_override"]
    np.testing.assert_array_equal(action, SentinelAgent().act(obs, KEY))


def test_uses_safe_half_sortie_instead_of_recalling_more_troops():
    obs = board([(0, 0, 107, True), (0, 1, 1, False)], [(3, 3, 4, True)], time=1)
    agent = SentinelV3Agent()
    memory = agent.initial_memory((4, 4))._replace(
        reserve=jnp.float32(10), defend_until=jnp.int32(10), last_turn=jnp.int32(0)
    )
    base = SentinelAgent().act(obs, KEY)
    assert tuple(np.asarray(base[1:3])) == (0, 0) and base[4] == 0
    action, _, telemetry = step(agent, obs, memory)
    np.testing.assert_array_equal(action, base.at[4].set(1))
    assert telemetry["defense_half_sortie"]
    assert not telemetry["defense_recall"]


def test_adequate_garrison_does_not_pull_in_extra_stacks():
    obs = board([(0, 0, 107, True), (0, 1, 2, False)], [(3, 3, 4, True)], time=1)
    agent = SentinelV3Agent()
    memory = agent.initial_memory((4, 4))._replace(
        reserve=jnp.float32(60), defend_until=jnp.int32(10), last_turn=jnp.int32(0)
    )
    action, _, telemetry = step(agent, obs, memory)
    assert action[0] == 1
    assert telemetry["defense_override"]
    assert not telemetry["defense_recall"]

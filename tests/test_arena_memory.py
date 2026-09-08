"""Stateful matches must agree with serial play without sharing player memory."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents import RandomAgent
from generals.core import game
from generals.core.observation import Observation
from generals.evaluation.arena import MatchResult, Rules, initial_memory, make_runner, transition
from generals.evaluation.cli import V3_OPTIONS, agent
from generals.evaluation.replay import make_policy, replay_game


class Memory(NamedTuple):
    turns: jax.Array
    seen: jax.Array
    random_sum: jax.Array


class RememberingPlayer:
    def initial_memory(self, shape):
        return Memory(jnp.int32(0), jnp.zeros(shape, jnp.int32), jnp.uint32(0))

    def step(self, obs, key, memory):
        assert isinstance(obs, Observation)
        action = RandomAgent().act(obs, key)
        action = jnp.where(memory.turns % 2 == 0, action, jnp.array([1, 0, 0, 0, 0]))
        next_memory = Memory(memory.turns + 1, memory.seen + obs.owned_cells, memory.random_sum + key[0])
        return action, next_memory, {"before": memory.turns, "nested": (obs.owned_land_count, key)}


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, e)


@pytest.mark.parametrize("seat", [0, 1])
@pytest.mark.parametrize("stateful_opponent", [False, True])
def test_stateful_arena_replay_and_serial_steps_agree(seat, stateful_opponent, tmp_path):
    grid = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 2]], dtype=np.int32)
    candidate = RememberingPlayer()
    opponent = RememberingPlayer() if stateful_opponent else RandomAgent().act
    rules = Rules(max_turns=12)
    seed = 391
    runner = make_runner(candidate, opponent, rules, with_memory=True)
    batch = runner(jnp.array(grid[None]), jax.random.PRNGKey(seed)[None], jnp.array([seat]))
    # Calling the same compiled runner again must create fresh episode memory.
    assert_tree_equal(batch, runner(jnp.array(grid[None]), jax.random.PRNGKey(seed)[None], jnp.array([seat])))
    arrays, result = replay_game(grid, candidate, opponent, rules, seat=seat, action_seed=seed)
    state = game.create_initial_state(jnp.array(grid))
    memories = [candidate.initial_memory(grid.shape), opponent.initial_memory(grid.shape) if stateful_opponent else ()]
    memory_histories = [[memories[0]], [memories[1]]]
    key = jax.random.PRNGKey(seed)
    for turn, recorded_actions in enumerate(arrays["actions"]):
        key, own_key, enemy_key = jax.random.split(key, 3)
        own_obs, enemy_obs = game.get_observation(state, seat), game.get_observation(state, 1 - seat)
        own, memories[0], _ = candidate.step(own_obs, own_key, memories[0])
        if stateful_opponent:
            enemy, memories[1], _ = opponent.step(enemy_obs, enemy_key, memories[1])
        else:
            enemy = opponent(enemy_obs, enemy_key)
        actions = jnp.stack((own, enemy) if seat == 0 else (enemy, own))
        np.testing.assert_array_equal(recorded_actions, actions)
        np.testing.assert_array_equal(arrays["telemetry_before"][turn], turn)
        for player in range(2):
            memory_histories[player].append(memories[player])
        state, _ = transition(state, actions, rules)
    assert_tree_equal(jax.tree.map(lambda x: x[0], batch.state), state)
    assert_tree_equal(batch.candidate_memory, jax.tree.map(lambda x: x[None], memories[0]))
    assert_tree_equal(batch.opponent_memory, jax.tree.map(lambda x: x[None], memories[1]))
    np.testing.assert_array_equal(batch.keys[0], key)
    np.testing.assert_array_equal(arrays["final_key"], key)
    np.testing.assert_array_equal(batch.counters[0], arrays["action_counters"].sum(axis=0))
    assert result["turns"] == int(batch.state.time[0])
    for player, role in enumerate(("candidate", "opponent")):
        for index in range(len(jax.tree.leaves(memories[player]))):
            np.testing.assert_array_equal(
                arrays[f"{role}_memory_leaf_{index}"],
                np.stack([jax.tree.leaves(memory)[index] for memory in memory_histories[player]]),
            )
    assert arrays["candidate_memory_paths"].tolist() == [".turns", ".seen", ".random_sum"]
    assert "['nested'][1]" in arrays["candidate_telemetry_paths"]
    # The trace remains readable without arbitrary-object pickle support.
    np.savez_compressed(tmp_path / "trace.npz", **arrays)
    with np.load(tmp_path / "trace.npz", allow_pickle=False) as saved:
        for name, expected in arrays.items():
            np.testing.assert_array_equal(saved[name], expected)


class AttackWithMemory(RememberingPlayer):
    def step(self, obs, key, memory):
        _, memory, telemetry = super().step(obs, key, memory)
        index = jnp.argmax((obs.owned_cells & ~obs.generals).reshape(-1))
        attack = jnp.array([0, index // 3, index % 3, 0, 0], dtype=jnp.int32)
        action = jnp.where(
            jnp.max(jnp.where(obs.owned_cells & ~obs.generals, obs.armies, 0)) > 5, attack, jnp.array([1, 0, 0, 0, 0])
        )
        return action, memory, telemetry


def test_memory_and_keys_freeze_for_terminal_draw_and_unequal_game_lengths():
    state = game.create_initial_state(jnp.array([[1, 0, 2], [0, 0, 0]], dtype=jnp.int32))
    state = state._replace(
        armies=state.armies.at[1, 0].set(10).at[1, 2].set(10),
        ownership=state.ownership.at[0, 1, 2].set(True).at[1, 1, 0].set(True),
        ownership_neutral=state.ownership_neutral.at[1, 0].set(False).at[1, 2].set(False),
    )
    quiet = state._replace(armies=state.armies.at[1, 0].set(1).at[1, 2].set(1))
    rules = Rules(max_turns=5, deathtouch_turn=0)
    ended, _ = transition(state, jnp.array([[0, 1, 2, 0, 0], [0, 1, 0, 0, 0]]), rules)
    states = jax.tree.map(lambda *values: jnp.stack(values), state, quiet, ended)
    keys = jax.random.split(jax.random.PRNGKey(71), 3)
    # The same object is deliberately used for both players.
    player = AttackWithMemory()
    run = make_runner(player, player, rules, from_states=True, with_memory=True)
    initial_finished = jnp.array([False, False, True])
    result = run(states, keys, jnp.array([0, 1, 0]), initial_finished)
    np.testing.assert_array_equal(result.state.time, [1, 5, 1])
    np.testing.assert_array_equal(result.finished, [True, False, True])
    np.testing.assert_array_equal(result.state.winner, [-1, -1, -1])
    for memory in (result.candidate_memory, result.opponent_memory):
        np.testing.assert_array_equal(memory.turns, [1, 5, 0])
    np.testing.assert_array_equal(result.keys[0], jax.random.split(keys[0], 3)[0])
    np.testing.assert_array_equal(result.keys[2], keys[2])
    assert not np.array_equal(result.candidate_memory.seen[1], result.opponent_memory.seen[1])
    assert not np.array_equal(result.candidate_memory.random_sum[1], result.opponent_memory.random_sum[1])
    np.testing.assert_array_equal(result.counters[2], np.zeros((2, 6), np.int32))
    # Finished games must match their independently run counterparts exactly.
    for index in range(3):
        individual = run(
            jax.tree.map(lambda x: x[index : index + 1], states),
            keys[index : index + 1],
            jnp.array([[0, 1, 0][index]]),
            initial_finished[index : index + 1],
        )
        assert_tree_equal(jax.tree.map(lambda x: x[index : index + 1], result), individual)


def test_stateless_default_result_stays_compatible():
    grid = jnp.array([[[1, 0], [0, 2]]], jnp.int32)
    key, seat = jax.random.PRNGKey(6)[None], jnp.array([1])
    player = RandomAgent().act
    plain = make_runner(player, player, Rules(max_turns=8))(grid, key, seat)
    extra = make_runner(player, player, Rules(max_turns=8), with_memory=True)(grid, key, seat)
    assert isinstance(plain, MatchResult)
    assert len(plain) == 3
    assert_tree_equal(plain, MatchResult(extra.state, extra.finished, extra.counters))
    assert extra.candidate_memory == extra.opponent_memory == ()


def test_incomplete_memory_contract_fails_clearly():
    class MissingStep:
        def initial_memory(self, shape):
            return ()

    with pytest.raises(TypeError, match="both initial_memory"):
        initial_memory(MissingStep(), (2, 2))


@pytest.mark.parametrize("name", list(V3_OPTIONS))
def test_v3_aliases_and_snapshots_preserve_rule_and_ablation_options(name, tmp_path):
    rules = Rules(max_turns=1200, build_castles=True, deathtouch_turn=800)
    snapshot = tmp_path / "snapshot.py"
    snapshot.write_text("from generals.agents.sentinel_v3_agent import SentinelV3Agent\n")
    direct = agent(name, rules)
    loaded, decision = make_policy(name, rules, source=snapshot)
    assert decision is None
    for player in (direct, loaded):
        for field, value in V3_OPTIONS[name].items():
            assert getattr(player, field) == value
        for field in ("max_turns", "build_castles", "deathtouch_turn"):
            assert getattr(player, field) == getattr(rules, field)
    # Explicit metadata supports historical runs that used the base v3 name
    # with ablation constructor arguments before aliases were introduced.
    historical, _ = make_policy("sentinel-v3", rules, options=V3_OPTIONS[name])
    assert historical.remember_threats == direct.remember_threats
    assert historical.sustained_defense == direct.sustained_defense


def test_real_v3_replay_matches_arena_and_records_telemetry():
    grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int32)
    rules = Rules(max_turns=6)
    candidate, opponent = agent("sentinel-v3", rules), RandomAgent().act
    result = make_runner(candidate, opponent, rules, with_memory=True)(
        jnp.array(grid[None]), jax.random.PRNGKey(41)[None], jnp.array([1])
    )
    arrays, actual = replay_game(grid, candidate, opponent, rules, seat=1, action_seed=41)
    assert actual["turns"] == int(result.state.time[0])
    assert actual["winner"] == int(result.state.winner[0])
    assert "telemetry_remembered_reserve" in arrays
    for index, leaf in enumerate(jax.tree.leaves(result.candidate_memory)):
        np.testing.assert_array_equal(arrays[f"candidate_memory_leaf_{index}"][-1], leaf[0])
    for field in result.state._fields:
        np.testing.assert_array_equal(arrays[f"state_{field}"][-1], getattr(result.state, field)[0])

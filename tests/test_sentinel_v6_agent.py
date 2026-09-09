"""Owned-defender commitment contracts, without claims of complete-game strength."""

import base64
import io
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v5_agent import SentinelV5Agent
from generals.agents.sentinel_v6_agent import DefenderMemory, SentinelV6Agent
from generals.core import game
from generals.evaluation.arena import Rules, transition
from tests.test_sentinel_agent import board
from tests.test_sentinel_v5_agent import _GAME24_WIRE, KEY, intercept_board, legal


def initial_state(obs):
    home = np.argwhere(np.asarray(obs.owned_cells & obs.generals))[0]
    enemies = np.argwhere(np.asarray(obs.opponent_cells & obs.generals))
    enemy = enemies[0] if len(enemies) else np.array([14, 5])
    grid = jnp.where(obs.mountains, -2, 0).at[tuple(home)].set(1).at[tuple(enemy)].set(2)
    return game.create_initial_state(grid)._replace(
        armies=obs.armies,
        ownership=jnp.stack((obs.owned_cells, obs.opponent_cells)),
        ownership_neutral=obs.neutral_cells,
        castles=obs.castles,
        time=obs.timestep,
    )


def game5_obs():
    return read_observation(io.StringIO(zlib.decompress(base64.b85decode(_GAME5_WIRE)).decode()), 18, 21)


def test_recorded_game5_arrival_is_not_recruited_back_and_forth():
    # Exact public t877 followed by reconstructed engine transitions under
    # recorded enemy actions. Unseen cells are empty, and a distant enemy
    # general is supplied; this is a local tactical check, not an exact replay.
    obs = game5_obs()
    state = initial_state(obs)
    agent = SentinelV6Agent(build_castles=True, deathtouch_turn=800)
    memory = agent.initial_memory(obs.armies.shape)
    enemy_actions = [[0, 2, 15, 3, 0], [0, 2, 16, 3, 0], [0, 2, 17, 3, 0], [0, 2, 18, 0, 0], [0, 1, 18, 0, 0]]
    actions, holds = [], []
    for index, enemy in enumerate(enemy_actions):
        obs = obs if index == 0 else game.get_observation(state, 0)
        action, memory, telemetry = agent.step(obs, KEY, memory)
        actions.append(np.asarray(action))
        holds.append(bool(telemetry["commitment_held"]))
        state, _ = transition(state, jnp.stack((action, jnp.array(enemy))), Rules(1200, True, 800))
    np.testing.assert_array_equal(actions[0], [0, 3, 14, 3, 0])
    assert any(holds[1:])
    assert not any(np.array_equal(a, [0, 3, 15, 2, 0]) for a in actions[1:])
    assert state.armies[3, 15] >= 37


def test_three_step_commitment_progresses_to_fixed_target_with_actual_transfers():
    corridor = {(r, 3) for r in range(7)} | {(4, 0), (4, 1), (4, 2)}
    walls = [(r, c) for r in range(7) for c in range(5) if (r, c) not in corridor]
    own = [(6, 3, 5, True), (4, 0, 10, False), (4, 1, 4, False), (4, 2, 3, False)]
    own += [(r, 3, 7 if r == 4 else 1, False) for r in range(1, 6)]
    obs = board(own, [(0, 3, 36, True)], mountains=walls, shape=(7, 5), time=232)
    state = initial_state(obs)
    agent = SentinelV6Agent()
    memory = agent.initial_memory(obs.armies.shape)
    for col in range(3):
        obs = game.get_observation(state, 0)
        action, memory, tel = agent.step(obs, KEY, memory)
        np.testing.assert_array_equal(action, [0, 4, col, 3, 0])
        legal(obs, action)
        assert memory.target == 23 and memory.distance == 2 - col
        assert bool(tel["commitment_started"]) == (col == 0)
        assert bool(tel["commitment_continued"]) == (col > 0)
        assert not tel["commitment_observation_gap"] and not tel["commitment_expired"]
        state, _ = game.step(state, jnp.stack((action, jnp.array([1, 0, 0, 0, 0]))))
    assert state.armies[4, 3] == 21


def test_actual_equal_eta_rescue_is_retained():
    obs = read_observation(io.StringIO(zlib.decompress(base64.b85decode(_GAME24_WIRE)).decode()), 21, 18)
    agent = SentinelV6Agent(build_castles=True, deathtouch_turn=800)
    action, memory, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    np.testing.assert_array_equal(action, [0, 16, 15, 3, 0])
    legal(obs, action)
    assert tel["intercept_enemy_eta"] == tel["intercept_friendly_eta"] == 1
    assert memory.expected_army == 13


@pytest.mark.parametrize("mutation", ["lost", "depleted", "gap", "expired", "clear", "grown"])
def test_invalid_or_obsolete_commitment_releases(mutation):
    obs = intercept_board()
    agent = SentinelV6Agent()
    action, memory, _ = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
    army = obs.armies.at[2, 2].set(1).at[2, 3].set(14)
    current = obs._replace(armies=army, timestep=obs.timestep + 1)
    if mutation == "lost":
        current = current._replace(owned_cells=current.owned_cells.at[2, 3].set(False))
    if mutation == "depleted":
        current = current._replace(armies=current.armies.at[2, 3].set(2))
    if mutation == "gap":
        current = current._replace(timestep=current.timestep + 1)
    if mutation == "expired":
        memory = memory._replace(expires=obs.timestep)
    if mutation == "clear":
        current = current._replace(opponent_cells=jnp.zeros_like(current.opponent_cells))
    if mutation == "grown":
        current = current._replace(armies=jnp.where(current.opponent_cells, 100, current.armies))
    _, new, tel = agent.step(current, KEY, memory)
    assert tel["commitment_released"]
    assert not tel["commitment_held"] and not tel["commitment_continued"]
    assert bool(tel["commitment_expired"]) == (mutation == "expired")
    assert bool(tel["commitment_observation_gap"]) == (mutation == "gap")
    if mutation in {"clear", "grown"}:
        assert new.defender == -1


def test_disabled_matches_v5_action_and_all_original_telemetry():
    agent = SentinelV6Agent(commit_defense=False)
    base = SentinelV5Agent()
    for obs in [intercept_board(), intercept_board(enemy=100, own=5), intercept_board(home=40)]:
        action, memory, tel = agent.step(obs, KEY, agent.initial_memory(obs.armies.shape))
        expected, old = base.decision(obs, KEY)
        np.testing.assert_array_equal(action, expected)
        assert tel.keys() == old.keys()
        for name in old:
            np.testing.assert_array_equal(tel[name], old[name])
        assert memory.defender == -1


def test_jit_vmap_memory_is_independent_and_fixed_dtype():
    agent = SentinelV6Agent()
    a, b = intercept_board(), intercept_board(home=40)
    observations = jax.tree.map(lambda x, y: jnp.stack((x, y)) if x is not None else None, a, b)
    memory = agent.initial_memory(a.armies.shape)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    actions, memories, tel = jax.jit(jax.vmap(agent.step))(observations, jax.random.split(KEY, 2), memories)
    assert actions.shape == (2, 5)
    assert memories.defender[0] >= 0 and memories.defender[1] == -1
    assert all(x.dtype == jnp.int32 for x in memories)


_GAME5_WIRE = (
    "c-qaBQF6r~2u1&01xo;oG2Q>hY7`?VHvQ?m4l_oR6A-w1<qU)oVopF2AtefB1cW{!P%)PAe%t27D%q({(nH=OD{^L5W6;t0iOe1y"
    "P`D%<tB8j~_MEIg*3V$G;oUC9XWyywebmfE9}KIi(o!W;J-14>OE&VT8>;BdY+SBZznI)RNvaRe_n_4=q9>!0$}B!-rSuNkeI^*a"
    "wH_Lo7Sgpv$(B!J9T47Z)smNsIx#_4(zE`T^(pWClM-xyP)K|sQO@Josd|PzH~kS-b+{g<da|kysSG_5xAA=?)0Ee)o8-)PB&$B`"
    "$#HV%)u&a#JLxI;6Xa1XJ7g5orzsiI>tY;byMlF>YPjCinCpFV)LSP{C<@w~(OJ<{7xPwQMdK5-%Y@-voVJ7-d5EoKez@kTl&4t9"
    "r`UXnot5jNJIeZ=UGU{zcf4Hq*N?)Jk~V77Q2w>-`A=nkNW=bv?7snwkY&*"
)


def active_memory(obs, defender, target=None, previous=None):
    return DefenderMemory(
        *(
            jnp.int32(x)
            for x in (
                defender,
                defender if target is None else target,
                defender if previous is None else previous,
                0,
                int(obs.timestep) + 5,
                int(obs.timestep) - 1,
                int(obs.armies.reshape(-1)[defender]),
            )
        )
    )


def test_fresh_guard_keeps_priority_over_an_arrived_commitment():
    open_cells = {(r, 3) for r in range(1, 6)} | {(2, 4), (2, 5)}
    walls = [(r, c) for r in range(6) for c in range(6) if (r, c) not in open_cells]
    obs = board(
        [(5, 3, 8, True), (2, 3, 20, False), (3, 3, 1, False), (4, 3, 1, False), (2, 4, 1, False)],
        [(1, 3, 20, False), (2, 5, 30, True)],
        mountains=walls,
        shape=(6, 6),
        time=232,
    )
    agent = SentinelV6Agent()
    expected, before = agent.base.decision(obs, KEY)
    assert before["intercept_guard"]
    action, memory, tel = agent.step(obs, KEY, active_memory(obs, 15, previous=16))
    np.testing.assert_array_equal(action, expected)
    legal(obs, action)
    assert tel["commitment_released"] and not tel["commitment_held"]
    assert memory.defender == -1


def test_winning_capture_releases_active_defender_commitment():
    obs = board([(5, 3, 3, True), (2, 2, 2, False)], [(1, 3, 50, False), (2, 3, 100, True)], shape=(6, 6), time=800)
    agent = SentinelV6Agent(deathtouch_turn=800)
    action, memory, tel = agent.step(obs, KEY, active_memory(obs, 14))
    np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    legal(obs, action)
    assert memory.defender == -1 and tel["commitment_released"]

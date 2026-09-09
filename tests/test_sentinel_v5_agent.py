"""Visible interception feasibility; complete games establish strategic strength."""

import base64
import io
import zlib

import jax
import jax.numpy as jnp
import numpy as np

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_agent import SentinelAgent
from generals.agents.sentinel_v5_agent import SentinelV5Agent, _home_growth
from generals.core import game
from generals.core.action import compute_valid_move_mask_obs
from tests.test_sentinel_agent import board

KEY = jax.random.PRNGKey(0)


def intercept_board(enemy=20, own=14, home=8, time=232):
    return board(
        [(5, 3, home, True), (2, 2, own, False), (2, 3, 1, False), (3, 3, 1, False), (4, 3, 1, False)],
        [(1, 3, enemy, False)],
        shape=(6, 6),
        time=time,
    )


def legal(obs, action):
    assert action[0] == 0
    row, col, direction = map(int, action[1:4])
    assert bool(compute_valid_move_mask_obs(obs)[row, col, direction])
    assert obs.armies[row, col] > 1


def test_forward_reinforcement_has_adequate_timely_added_defense():
    obs = intercept_board()
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    legal(obs, action)
    assert telemetry["intercept_override"] and telemetry["intercept_feasible"]
    assert telemetry["intercept_friendly_eta"] <= telemetry["intercept_enemy_eta"]
    assert telemetry["intercept_new_defense"] >= telemetry["intercept_required_defense"] > 0
    assert not obs.generals[int(action[1]), int(action[2])]


def test_no_futile_recall_when_no_stack_can_add_enough_defense_in_time():
    obs = intercept_board(enemy=100, own=5)
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    np.testing.assert_array_equal(action, SentinelAgent().act(obs, KEY))
    assert not telemetry["intercept_override"]
    assert not telemetry["intercept_feasible"]
    assert telemetry["intercept_home_deficit"] > 50


def test_existing_corridor_garrisons_are_not_counted_twice_as_new_defense():
    corridor = {(r, 3) for r in range(1, 6)}
    walls = [(r, c) for r in range(6) for c in range(6) if (r, c) not in corridor]
    obs = board(
        [(5, 3, 3, True), (2, 3, 10, False), (3, 3, 1, False), (4, 3, 1, False)],
        [(1, 3, 24, False)],
        mountains=walls,
        shape=(6, 6),
        time=232,
    )
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    assert not telemetry["intercept_feasible"]
    np.testing.assert_array_equal(action, SentinelAgent().act(obs, KEY))


def test_three_step_owned_collection_matches_actual_transfers_without_counting_screen_twice():
    corridor = {(r, 3) for r in range(7)} | {(4, 0), (4, 1), (4, 2)}
    walls = [(r, c) for r in range(7) for c in range(5) if (r, c) not in corridor]
    owned = [(6, 3, 5, True), (4, 0, 10, False), (4, 1, 4, False), (4, 2, 3, False)]
    owned += [(r, 3, 7 if r == 4 else 1, False) for r in range(1, 6)]
    obs = board(owned, [(0, 3, 36, True)], mountains=walls, shape=(7, 5), time=232)
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    np.testing.assert_array_equal(action, [0, 4, 0, 3, 0])
    assert telemetry["intercept_target_index"] == 4 * 5 + 3
    assert telemetry["intercept_friendly_eta"] == 3
    assert telemetry["intercept_enemy_eta"] == 4
    # Only off-corridor donors add defense: (10-1)+(4-1)+(3-1)=14.
    # The target's pre-existing seven already contribute stationary attrition.
    assert telemetry["intercept_new_defense"] == 14
    assert telemetry["intercept_arrival_army"] == 21
    grid = jnp.where(obs.mountains, -2, 0).at[6, 3].set(1).at[0, 3].set(2)
    state = game.create_initial_state(grid)._replace(
        armies=obs.armies,
        ownership=jnp.stack((obs.owned_cells, obs.opponent_cells)),
        ownership_neutral=obs.neutral_cells,
        time=obs.timestep,
    )
    for col in (0, 1, 2):
        own_action = jnp.array([0, 4, col, 3, 0], jnp.int32)
        legal(game.get_observation(state, 0), own_action)
        state, _ = game.step(state, jnp.stack((own_action, jnp.array([1, 0, 0, 0, 0]))))
    assert state.armies[4, 3] == telemetry["intercept_arrival_army"]
    assert bool(state.ownership[0, 4, 3])


def test_benign_enemy_and_original_corridor_controls_keep_campaign_actions():
    agent, base = SentinelV5Agent(), SentinelAgent()
    observations = [
        intercept_board(home=40),
        board([(1, 1, 2, True)], [(2, 1, 2, True)]),
        board([(3, 1, 8, True)], [(2, 2, 8, True)], mountains=[(3, 0), (3, 2)]),
    ]
    for obs in observations:
        np.testing.assert_array_equal(agent.act(obs, KEY), base.act(obs, KEY))


def test_guaranteed_home_growth_counts_only_before_attack():
    np.testing.assert_array_equal(_home_growth(jnp.int32(49), jnp.array([1, 2, 3, 4])), [0, 2, 2, 3])
    np.testing.assert_array_equal(_home_growth(jnp.int32(50), jnp.array([1, 2, 3, 4])), [0, 0, 1, 1])
    # Force a ten-step corridor; four apparent missing defenders are supplied
    # by guaranteed home growth, so no buffer-driven withdrawal is justified.
    open_cells = {(r, 2) for r in range(11)} | {(8, 1), (8, 0)}
    mountains = [(r, c) for r in range(11) for c in range(4) if (r, c) not in open_cells]
    obs = board([(10, 2, 10, True), (8, 1, 9, False)], [(0, 2, 24, False)], mountains=mountains, shape=(11, 4), time=1)
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    np.testing.assert_array_equal(action, SentinelAgent().act(obs, KEY))
    assert not telemetry["intercept_override"]


def test_currently_adequate_screen_cannot_be_drained_into_a_side_branch():
    open_cells = {(r, 3) for r in range(1, 6)} | {(2, 4), (2, 5)}
    walls = [(r, c) for r in range(6) for c in range(6) if (r, c) not in open_cells]
    obs = board(
        [(5, 3, 8, True), (2, 3, 20, False), (3, 3, 1, False), (4, 3, 1, False), (2, 4, 1, False)],
        [(1, 3, 20, False), (2, 5, 30, True)],
        mountains=walls,
        shape=(6, 6),
        time=232,
    )
    base = SentinelAgent().act(obs, KEY)
    np.testing.assert_array_equal(base, [0, 2, 3, 3, 0])
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    assert telemetry["intercept_guard"] and telemetry["intercept_override"]
    assert not np.array_equal(action, base)
    legal(obs, action)


def test_expensive_already_safe_branch_is_not_selected_as_interception():
    obs = board(
        [(5, 3, 1, True), (2, 3, 30, False), (2, 2, 5, False)],
        [(0, 3, 25, False)],
        shape=(6, 6),
        time=231,
    )
    action, telemetry = SentinelV5Agent().decision(obs, KEY)
    assert not telemetry["intercept_feasible"]
    assert telemetry["intercept_target_index"] == -1
    np.testing.assert_array_equal(action, SentinelAgent().act(obs, KEY))


def test_projected_deathtouch_removes_home_army_from_future_protection():
    obs = intercept_board(home=40, time=799)
    _, ordinary = SentinelV5Agent().decision(obs, KEY)
    action, projected = SentinelV5Agent(deathtouch_turn=800).decision(obs, KEY)
    assert not ordinary["intercept_override"]
    assert projected["intercept_projected_deathtouch"]
    assert projected["intercept_home_deficit"] > 0
    assert not projected["intercept_feasible"]  # A small blocker cannot stop this army before touch.
    np.testing.assert_array_equal(action, SentinelAgent(deathtouch_turn=800).act(obs, KEY))


def test_winning_general_capture_precedes_visible_home_threat():
    obs = board([(5, 3, 3, True), (2, 2, 2, False)], [(1, 3, 50, False), (2, 3, 100, True)], shape=(6, 6), time=800)
    action, telemetry = SentinelV5Agent(deathtouch_turn=800).decision(obs, KEY)
    np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    assert not telemetry["intercept_override"]


def test_disabled_actions_and_original_telemetry_match_v2_and_enabled_vmaps():
    size, shape = 16, (6, 6)
    grid = jnp.zeros(shape, jnp.int32).at[0, 0].set(1).at[5, 5].set(2)
    states = jax.vmap(game.create_initial_state)(jnp.broadcast_to(grid, (size,) + shape))
    rng = np.random.default_rng(95307)
    owners = rng.integers(0, 3, (size,) + shape)
    owners[:, 0, 0], owners[:, 5, 5] = 0, 1
    states = states._replace(
        ownership=jnp.array(np.stack((owners == 0, owners == 1), axis=1)),
        ownership_neutral=jnp.array(owners == 2),
        armies=jnp.array(rng.integers(1, 90, (size,) + shape, dtype=np.int32)),
        time=jnp.array(rng.choice([0, 49, 232, 799, 800, 1150], size), jnp.int32),
    )
    observations = jax.vmap(lambda state: game.get_observation(state, 0))(states)
    keys = jax.random.split(KEY, size)
    kwargs = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
    reference = jax.jit(jax.vmap(SentinelAgent(**kwargs).decision))(observations, keys)
    actions, telemetry = jax.jit(jax.vmap(SentinelV5Agent(**kwargs, intercept_threats=False).decision))(
        observations, keys
    )
    np.testing.assert_array_equal(actions, reference[0])
    for name, value in reference[1].items():
        np.testing.assert_array_equal(telemetry[name], value)
    enabled = SentinelV5Agent(**kwargs)
    batched = jax.jit(jax.vmap(enabled.act))(observations, keys)
    for index in (0, 7, 15):
        obs = jax.tree.map(lambda value: value[index], observations)
        np.testing.assert_array_equal(batched[index], enabled.act(obs, keys[index]))


def test_recorded_equal_eta_forward_block_preserves_ordinary_home():
    # Exact public game24/t232/seat0 wire from the consumed Amin development
    # campaign. Numeric protocol fixture is embedded for clean-checkout tests.
    wire = zlib.decompress(base64.b85decode(_GAME24_WIRE)).decode()
    obs = read_observation(io.StringIO(wire), 21, 18)
    action, telemetry = SentinelV5Agent(build_castles=True, deathtouch_turn=800).decision(obs, KEY)
    np.testing.assert_array_equal(action, [0, 16, 15, 3, 0])
    legal(obs, action)
    assert telemetry["intercept_enemy_eta"] == telemetry["intercept_friendly_eta"] == 1
    assert telemetry["intercept_new_defense"] == 12
    assert telemetry["intercept_required_defense"] > 0
    assert telemetry["intercept_arrival_army"] == 13


_GAME24_WIRE = (
    "c-rNaQL@7z2u1%{1xpYN({}$GdnLq%2pWI-^Eym3YK|AYGAUQa?TJ(c9|h@Ap`e8vzxH&iAeVW`W}vgM(^2myT=k=>>^h@N9|~+m"
    "j*M5b)qL&}^kWM~LV~OIN!t!`Oy*5wG*o$4cOzx|ep{>U-G}iRx_3<pqA#A{P)|D#ceyfuTiu+Tt8{f8EX4bizIrHw{&8B6^+o3c"
    "ME^ab6>X3cGQ6)r{7Z)S`<e%RiQ&-EG}O^I=V+A6Q!{3s>(rsCKSn7t^EvDJoGA;;O0WKMKd&$={_vL=-s6>i=i?WOkI^=%`Tg&G"
    "doE=xW(?ovYSe}-Q=>H@)f=a+s&d0vEIG|f?ePcOG-Fx"
)

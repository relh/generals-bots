"""Classic move order and transition parity with the pinned Softmax engine."""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.core import coworld_game, game
from generals.core.env import GeneralsEnv
from integrations.softmax.engine import Match


def board():
    grid = jnp.zeros((5, 5), jnp.int32).at[0, 0].set(1).at[4, 4].set(2)
    return game.create_initial_state(grid)


def claim(state, player, row, col, army):
    return state._replace(
        ownership=state.ownership.at[:, row, col].set(jnp.arange(2) == player),
        ownership_neutral=state.ownership_neutral.at[row, col].set(False),
        armies=state.armies.at[row, col].set(army),
    )


def test_pinned_engine_source():
    assert hashlib.sha256(Path(coworld_game.__file__).read_bytes()).hexdigest() == (
        "f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318"
    )


@pytest.mark.parametrize("time, expected", [(0, [0, 1]), (1, [1, 0])])
def test_equal_army_ties_reverse_on_odd_turns(time, expected):
    state = claim(claim(board(), 0, 2, 1, 10), 1, 2, 3, 10)
    state = state._replace(time=jnp.asarray(time, jnp.int32), armies=state.armies.at[2, 2].set(1))
    actions = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 3, 2, 0]], jnp.int32)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, actions), expected)
    final, _ = coworld_game.step(state, actions)
    # Last equal attacker captures the contested neutral tile.
    assert bool(final.ownership[expected[-1], 2, 2])


def test_larger_army_resolves_first():
    state = claim(claim(board(), 0, 2, 1, 10), 1, 2, 3, 5)
    state = state._replace(armies=state.armies.at[2, 2].set(1))
    actions = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 3, 2, 0]], jnp.int32)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, actions), [0, 1])
    final, _ = coworld_game.step(state, actions)
    assert int(final.armies[2, 2]) == 4
    # The old training rule produces a different combat result.
    old, _ = game.step(state, actions)
    assert int(old.armies[2, 2]) == 6


def test_owned_land_merge_precedes_larger_attack():
    state = claim(claim(claim(board(), 0, 2, 1, 5), 0, 2, 2, 3), 1, 2, 3, 100)
    actions = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 3, 2, 0]], jnp.int32)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, actions), [0, 1])


def test_general_attack_resolves_after_ordinary_attack():
    state = claim(claim(board(), 0, 3, 4, 5), 1, 2, 2, 100)
    actions = jnp.asarray([[0, 3, 4, 1, 0], [0, 2, 2, 3, 0]], jnp.int32)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, actions), [1, 0])


def test_chased_move_waits_but_head_on_swap_uses_army_priority():
    state = claim(claim(board(), 0, 2, 1, 5), 1, 2, 2, 10)
    chase = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 2, 3, 0]], jnp.int32)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, chase), [0, 1])
    swap = chase.at[1, 3].set(2)
    np.testing.assert_array_equal(coworld_game._determine_move_order(state, swap), [1, 0])


def test_classic_replay_match_uses_hosted_head_on_order():
    state = claim(claim(board(), 0, 2, 1, 5), 1, 2, 2, 10)
    actions = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 2, 2, 1]], jnp.int32)
    expected, _ = coworld_game.step(state, actions)
    generic, _ = game.step(state, actions)
    assert int(expected.armies[2, 2]) != int(generic.armies[2, 2])

    match = Match(7, coworld_classic_rules=True)
    match.state = state
    match.height = match.width = 5
    match.advance(np.asarray(actions).tolist())
    for actual, wanted in zip(jax.tree.leaves(match.state), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_classic_environment_uses_official_transitions_and_observations():
    env = GeneralsEnv(grid_dims=(5, 5), pool_size=1, coworld_classic_rules=True)
    state = claim(claim(board(), 0, 2, 1, 10), 1, 2, 3, 5)
    pool = jax.tree.map(lambda x: x[None], state)
    actions = jnp.asarray([[0, 2, 1, 3, 0], [0, 2, 3, 2, 0]], jnp.int32)
    expected, _ = coworld_game.step(state, actions)
    timestep, actual = env.step(state, actions, pool)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(jax.tree.leaves(coworld_game.get_observations(expected)),
                           jax.tree.leaves(timestep.observation), strict=True):
        np.testing.assert_array_equal(left, right)


def test_classic_rules_reject_conflicting_modes():
    with pytest.raises(ValueError, match="capture-only"):
        GeneralsEnv(coworld_classic_rules=True, legacy_move_priority=True)
    with pytest.raises(ValueError, match="capture-only"):
        GeneralsEnv(coworld_classic_rules=True, build_castles=True)
    with pytest.raises(ValueError, match="two players"):
        GeneralsEnv(coworld_classic_rules=True, num_players=4)

"""Checks for paired game setup, outcome accounting, and batched simulation."""
import pytest

pytest.importorskip('equinox')
pytest.importorskip('optax')

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents import HunterAgent
from generals.core import game

from .evaluate import connected, make_boards, make_cases, make_runner, policy_action, summarize
from .network import PolicyValueNetwork


def test_cases_balance_position_and_player_id():
    boards = make_boards('open', 0)
    assert len(boards) == 120
    cases = make_cases(boards[:1], seed=1, repeats=2)
    assert len(cases) == 8
    for repeat in range(2):
        group = [case for case in cases if case['repeat'] == repeat]
        assignments = {(c['seat'], tuple(np.argwhere(c['grid'] == c['seat'] + 1)[0])) for c in group}
        assert len(assignments) == 4
        assert len({c['action_seed'] for c in group}) == 1
    terrain = make_boards('terrain', 12, terrain_boards=10)
    assert all(connected(board) for board in terrain)
    assert all(np.sum(board == -2) == 2 and np.sum(board > 2) == 2 for board in terrain)


def test_summary_keeps_draws_in_denominator():
    rows = [dict(board_id=i // 2, seat=i % 2, result=result, score=score, turns=10)
            for i, (result, score) in enumerate([('win', 1), ('loss', 0), ('draw', .5), ('draw', .5)])]
    result = summarize(rows)
    assert result['win_rate'] == .25
    assert result['score'] == .5
    assert result['draws'] == 2


def test_batched_games_match_serial_reference():
    network = PolicyValueNetwork(jax.random.PRNGKey(0), channels=(4, 4, 4, 4))
    opponent = HunterAgent()
    cases = make_cases(make_boards('open', 0)[:1], 9, repeats=1)
    grids = jnp.asarray(np.stack([case['grid'] for case in cases]))
    keys = jnp.stack([jax.random.PRNGKey(c['action_seed']) for c in cases])
    seats = jnp.array([c['seat'] for c in cases])
    actual = make_runner(opponent.act, max_turns=20)(network, grids, keys, seats)
    for i, case in enumerate(cases):
        state = game.create_initial_state(grids[i])
        key = keys[i]
        seat = case['seat']
        while int(state.winner) < 0 and int(state.time) < 20:
            key, candidate_key, opponent_key = jax.random.split(key, 3)
            ours = policy_action(network, game.get_observation(state, seat), candidate_key)
            theirs = opponent.act(game.get_observation(state, 1-seat), opponent_key)
            actions = jnp.stack([ours, theirs] if seat == 0 else [theirs, ours])
            state, _ = game.step(state, actions)
        for batched, single in zip(jax.tree.leaves(actual), jax.tree.leaves(state)):
            np.testing.assert_array_equal(batched[i], single)

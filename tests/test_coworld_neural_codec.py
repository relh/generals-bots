"""Canonical serving must reproduce the pinned Classic public view and actions."""

import jax.numpy as jnp
import numpy as np
import pytest

from generals.core import coworld_game
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import decode_policy_action, encode_wire_observation, training_observation


@pytest.mark.parametrize('height,width', [(18, 21), (21, 18), (19, 20), (21, 21)])
def test_wire_reconstructs_actual_public_view_with_padding_and_scalar_planes(height, width):
    grid = np.zeros((height, width), np.int32)
    grid[0, 0], grid[-1, -1], grid[1, 1], grid[2, 2] = 1, 2, 42, -2
    match = Match.__new__(Match)
    match._game = coworld_game
    match.state = coworld_game.create_initial_state(jnp.asarray(grid))
    match.height, match.width = height, width
    match.last_move_executed = [None, None]
    padded = np.pad(grid, ((0, 21 - height), (0, 21 - width)), constant_values=-2)
    padded_state = coworld_game.create_initial_state(jnp.asarray(padded))
    for seat in (0, 1):
        message = match.observation(seat)
        restored = training_observation(message)
        expected = coworld_game.get_observation(padded_state, seat)
        for name in expected._fields:
            if name.startswith("allied_"):
                continue  # The Classic wire carries the two opposing players.
            np.testing.assert_array_equal(getattr(restored, name), getattr(expected, name))
        values, mask = encode_wire_observation(message)
        training_values, training_mask = encode_coworld_directional_observation(expected)
        np.testing.assert_array_equal(values, training_values)
        np.testing.assert_array_equal(mask, training_mask)
        assert values.shape == (7056,) and values.dtype == np.float32
        assert mask.shape == (3529,) and mask.dtype == bool
        np.testing.assert_array_equal(mask[:1764], mask[1764:3528])
        assert mask[3528]
        for index in np.flatnonzero(mask[:3528]):
            action = np.asarray(decode_action(int(index)))
            assert 0 <= action[1] < height and 0 <= action[2] < width
        for turn in (0, 100, 1999):
            timed, timed_mask = encode_wire_observation(dict(message, turn=turn))
            np.testing.assert_array_equal(timed[:4851], values[:4851])
            np.testing.assert_allclose(timed.reshape(16, 441)[11], turn / 2000)
            np.testing.assert_array_equal(timed_mask, mask)


def test_flat_categorical_decoding_matches_training_for_full_half_and_pass():
    rng = np.random.default_rng(7)
    for cell in (0, 220, 440):
        for direction in range(4):
            for half in (0, 1):
                index = (half * 4 + direction) * 441 + cell
                probabilities = np.zeros(3529, np.float32)
                probabilities[index] = 1
                np.testing.assert_array_equal(decode_policy_action(probabilities, rng=rng), decode_action(index))
    probabilities = np.zeros(3529, np.float32)
    probabilities[3528] = 1
    assert decode_policy_action(probabilities, rng=rng) == [1, 0, 0, 0, 0]


def test_decoder_samples_the_actual_distribution_and_rejects_invalid_shapes():
    probabilities = np.zeros(3529, np.float32)
    probabilities[[0, 1764]] = [0.25, 0.75]
    rng = np.random.default_rng(123)
    expected = np.random.default_rng(123)
    for _ in range(32):
        index = expected.choice(3529, p=probabilities / probabilities.sum())
        np.testing.assert_array_equal(decode_policy_action(probabilities, rng=rng), decode_action(index))
    for invalid in (np.ones(1767), np.zeros(3529), np.full(3529, -1), np.full(3529, np.nan)):
        with pytest.raises(ValueError, match='3529-action'):
            decode_policy_action(invalid, rng=rng)
    with pytest.raises(ValueError, match='3529-action'):
        decode_policy_action(probabilities, rng=None)
    with pytest.raises(ValueError, match='21×21'):
        decode_action(0, 10)

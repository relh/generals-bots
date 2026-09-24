"""The hosted wire view must reproduce the neural policy's padded input."""

import jax.numpy as jnp
import numpy as np

from generals.core import game
from integrations.puffer_codec import (
    encode_coworld_directional_observation,
    encode_coworld_hinted_observation,
    encode_coworld_lean_observation,
    encode_coworld_observation,
    encode_coworld_packed_directional_observation,
    encode_observation,
)
from integrations.softmax.engine import Match
from integrations.softmax.neural_codec import encode_wire_observation, training_observation


def test_coworld_wire_view_matches_padded_training_view():
    height, width = 18, 20
    grid = np.zeros((height, width), dtype=np.int32)
    grid[0, 0] = 1
    grid[-1, -1] = 2
    grid[1, 1] = 42
    grid[2, 2] = -2
    padded = np.pad(grid, ((0, 21 - height), (0, 21 - width)), constant_values=-2)
    match = Match.__new__(Match)
    match.state = game.create_initial_state(jnp.asarray(grid))
    match.height, match.width = height, width
    match.last_move_executed = [None, None]

    message = match.observation(0)
    restored = training_observation(message)
    expected = game.get_observation(game.create_initial_state(jnp.asarray(padded)), 0)
    for name in (
        "armies", "generals", "castles", "mountains", "neutral_cells", "owned_cells", "opponent_cells",
        "fog_cells", "structures_in_fog", "owned_land_count", "owned_army_count", "opponent_land_count",
        "opponent_army_count", "timestep",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(restored, name)), np.asarray(getattr(expected, name)))

    values, mask = encode_wire_observation(message)
    expected_values, expected_mask = encode_observation(expected, factorized_actions=True, goal_features=True)
    np.testing.assert_array_equal(values, np.asarray(expected_values))
    np.testing.assert_array_equal(mask, np.asarray(expected_mask))
    assert values.shape == (9261,)
    assert mask.shape == (1767,)

    compact_values, compact_mask = encode_wire_observation(message, compact=True)
    expected_compact, expected_compact_mask = encode_coworld_observation(expected)
    np.testing.assert_array_equal(compact_values, np.asarray(expected_compact))
    np.testing.assert_array_equal(compact_mask, np.asarray(expected_compact_mask))
    assert compact_values.shape == (6174,)

    lean_values, lean_mask = encode_wire_observation(message, lean=True)
    expected_lean, expected_lean_mask = encode_coworld_lean_observation(expected)
    np.testing.assert_array_equal(lean_values, np.asarray(expected_lean))
    np.testing.assert_array_equal(lean_mask, np.asarray(expected_lean_mask))
    assert lean_values.shape == (3528,)

    directional_values, directional_mask = encode_wire_observation(message, directional=True)
    expected_directional, expected_directional_mask = encode_coworld_directional_observation(expected)
    np.testing.assert_array_equal(directional_values, np.asarray(expected_directional))
    np.testing.assert_array_equal(directional_mask, np.asarray(expected_directional_mask))
    assert directional_values.shape == (4851,)

    packed_values, packed_mask = encode_wire_observation(message, packed_directional=True)
    expected_packed, expected_packed_mask = encode_coworld_packed_directional_observation(expected)
    np.testing.assert_array_equal(packed_values, np.asarray(expected_packed))
    np.testing.assert_array_equal(packed_mask, np.asarray(expected_packed_mask))
    assert packed_values.shape == (3528,)

    hinted_values, hinted_mask = encode_wire_observation(message, hinted=True)
    expected_hinted, expected_hinted_mask = encode_coworld_hinted_observation(expected)
    np.testing.assert_array_equal(hinted_values, np.asarray(expected_hinted))
    np.testing.assert_array_equal(hinted_mask, np.asarray(expected_hinted_mask))
    assert hinted_values.shape == (3528,)

    prior_values, prior_mask = encode_wire_observation(message, prior_hinted=True)
    expected_prior, expected_prior_mask = encode_coworld_hinted_observation(expected, signed_flags=True)
    np.testing.assert_array_equal(prior_values, np.asarray(expected_prior))
    np.testing.assert_array_equal(prior_mask, np.asarray(expected_prior_mask))

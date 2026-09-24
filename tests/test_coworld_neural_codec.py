"""The hosted wire view must reproduce the neural policy's padded input."""

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.harvester_agent import expander_harvester_action
from generals.core import game
from generals.core.observation import Observation
from integrations.puffer_codec import (
    encode_coworld_directional_observation,
    encode_coworld_hinted_observation,
    encode_coworld_lean_observation,
    encode_coworld_observation,
    encode_coworld_packed_directional_observation,
    encode_observation, hinted_replay_indices,
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

    sprint_values, sprint_mask = encode_wire_observation(message, sprint_prior_hinted=True)
    expected_sprint, expected_sprint_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, sprint_hint=True
    )
    np.testing.assert_array_equal(sprint_values, np.asarray(expected_sprint))
    np.testing.assert_array_equal(sprint_mask, np.asarray(expected_sprint_mask))

    expander_values, expander_mask = encode_wire_observation(message, expander_prior_hinted=True)
    expected_expander, expected_expander_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, expander_hint=True
    )
    np.testing.assert_array_equal(expander_values, np.asarray(expected_expander))
    np.testing.assert_array_equal(expander_mask, np.asarray(expected_expander_mask))


def test_signed_hint_replay_metadata_covers_pass_and_move():
    planes = np.zeros((2, 8, 21 * 21), dtype=np.float32)
    planes[0, 3] = 1
    planes[0, 2] = -1
    planes[1, 3] = -1
    planes[1, 2] = 1
    planes[1, 5, 123] = 1
    np.testing.assert_array_equal(
        hinted_replay_indices(planes.reshape(2, -1), 21),
        np.asarray([[1764, -1], [441 + 123, 1]], dtype=np.int32),
    )


def test_capture_hint_rallies_surplus_for_reachable_castle():
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 0], armies[2, 1], armies[2, 2], armies[2, 3] = 3, 6, 2, 5
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, :3] = True
    general = np.zeros_like(owned)
    general[2, 0] = True
    castles = np.zeros_like(owned)
    castles[2, 3] = True
    neutral = ~owned
    empty = np.zeros_like(owned)
    observation = Observation(
        armies=jnp.asarray(armies), generals=jnp.asarray(general), castles=jnp.asarray(castles),
        mountains=jnp.asarray(empty), neutral_cells=jnp.asarray(neutral), owned_cells=jnp.asarray(owned),
        opponent_cells=jnp.asarray(empty), fog_cells=jnp.asarray(empty), structures_in_fog=jnp.asarray(empty),
        owned_land_count=jnp.int32(3), owned_army_count=jnp.int32(11),
        opponent_land_count=jnp.int32(1), opponent_army_count=jnp.int32(1), timestep=jnp.int32(200),
    )
    # The root cannot yet take the five-army city; one connected source can feed it.
    np.testing.assert_array_equal(
        np.asarray(expander_harvester_action(jax.random.PRNGKey(0), observation)),
        np.asarray([0, 2, 1, 3, 0]),
    )


def test_capture_hint_uses_home_general_without_visible_threat():
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 2] = 50
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, 2] = True
    empty = np.zeros_like(owned)
    observation = Observation(
        armies=jnp.asarray(armies), generals=jnp.asarray(owned), castles=jnp.asarray(empty),
        mountains=jnp.asarray(empty), neutral_cells=jnp.asarray(~owned), owned_cells=jnp.asarray(owned),
        opponent_cells=jnp.asarray(empty), fog_cells=jnp.asarray(empty), structures_in_fog=jnp.asarray(empty),
        owned_land_count=jnp.int32(1), owned_army_count=jnp.int32(50),
        opponent_land_count=jnp.int32(1), opponent_army_count=jnp.int32(1), timestep=jnp.int32(150),
    )
    assert int(expander_harvester_action(jax.random.PRNGKey(0), observation)[0]) == 0
    assert int(expander_harvester_action(
        jax.random.PRNGKey(0), observation._replace(timestep=jnp.int32(50))
    )[0]) == 0


def test_capture_hint_reinforces_home_against_nearby_enemy_stack():
    armies = np.zeros((7, 7), dtype=np.int32)
    armies[3, 2], armies[3, 3], armies[3, 5] = 20, 5, 15
    owned = np.zeros((7, 7), dtype=bool)
    owned[3, 2:4] = True
    general = np.zeros_like(owned)
    general[3, 3] = True
    enemy = np.zeros_like(owned)
    enemy[3, 5] = True
    empty = np.zeros_like(owned)
    observation = Observation(
        armies=jnp.asarray(armies), generals=jnp.asarray(general), castles=jnp.asarray(empty),
        mountains=jnp.asarray(empty), neutral_cells=jnp.asarray(~(owned | enemy)),
        owned_cells=jnp.asarray(owned), opponent_cells=jnp.asarray(enemy),
        fog_cells=jnp.asarray(empty), structures_in_fog=jnp.asarray(empty),
        owned_land_count=jnp.int32(2), owned_army_count=jnp.int32(25),
        opponent_land_count=jnp.int32(1), opponent_army_count=jnp.int32(15),
        timestep=jnp.int32(150),
    )
    np.testing.assert_array_equal(
        np.asarray(expander_harvester_action(jax.random.PRNGKey(0), observation)),
        np.asarray([0, 3, 2, 3, 0]),
    )

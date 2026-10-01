"""The hosted wire view must reproduce the neural policy's padded input."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.parametrize("height,width", [(18, 21), (21, 18), (19, 20), (21, 21)])
def test_calibrated_context_wire_matches_training_and_preserves_teacher_actions(height, width, tmp_path):
    from metta_training.environment import EnvironmentContext

    from integrations.metta_puffer import GeneralsPufferEnvironment

    kinds = np.ones((height, width), dtype=np.int32)
    owners = np.zeros((height, width), dtype=np.int32)
    armies = np.zeros((height, width), dtype=np.int32)
    kinds[2, 2], owners[2, 2], armies[2, 2] = 4, 1, 20
    owners[2, 3], armies[2, 3] = 1, 10
    kinds[-3, -3], owners[-3, -3], armies[-3, -3] = 4, 2, 20
    message = dict(
        height=height,
        width=width,
        type_grid=kinds.tolist(),
        owner_grid=owners.tolist(),
        army_grid=armies.tolist(),
        my_land=2,
        my_army=30,
        opp_land=1,
        opp_army=20,
        turn=50,
    )
    options = dict(
        coworld_classic=True,
        teacher="expander_harvester",
        factorized_actions=True,
        hint_features=True,
        prior_hint_features=True,
        expander_hint_features=True,
        context_hint_features=True,
        compact_features=True,
        lean_features=True,
        move_hint_scale=0.375,
        split_hint_scale=0.125,
    )
    env = GeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1322, index=0, mode="train", output=tmp_path),
        **options,
    )
    try:
        training_values, training_mask = env._encode(training_observation(message))
        wire_values, wire_mask = encode_wire_observation(
            message,
            expander_context_prior_hinted=True,
            move_hint_scale=0.375,
            split_hint_scale=0.125,
        )
        original, original_mask = encode_wire_observation(message, expander_context_prior_hinted=True)
        np.testing.assert_array_equal(wire_values, np.asarray(training_values))
        np.testing.assert_array_equal(wire_mask, np.asarray(training_mask))
        np.testing.assert_array_equal(wire_mask, original_mask)
        np.testing.assert_array_equal(
            hinted_replay_indices(wire_values, 21, 14),
            hinted_replay_indices(original, 21, 14),
        )
        np.testing.assert_array_equal(wire_values.reshape(14, 441)[8:], original.reshape(14, 441)[8:])
    finally:
        env.close()


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

    flat_values, flat_mask = encode_wire_observation(message, directional=True, factorized_actions=False)
    expected_flat_values, expected_flat_mask = encode_coworld_directional_observation(
        expected, factorized_actions=False,
    )
    np.testing.assert_array_equal(flat_values, np.asarray(expected_flat_values))
    np.testing.assert_array_equal(flat_mask, np.asarray(expected_flat_mask))
    assert flat_values.shape == (4851,)
    assert flat_mask.shape == (3529,)

    for turn in (0, 25, 100, 1199):
        timed_values, timed_mask = encode_wire_observation(
            dict(message, turn=turn), directional=True, factorized_actions=False,
            directional_time_features=True,
        )
        expected_timed, expected_timed_mask = encode_coworld_directional_observation(
            expected._replace(timestep=jnp.int32(turn)), factorized_actions=False, include_timestep=True,
        )
        np.testing.assert_array_equal(timed_values, np.asarray(expected_timed))
        np.testing.assert_array_equal(timed_values[:4851], flat_values)
        np.testing.assert_allclose(timed_values[4851:], turn / 1200.0)
        np.testing.assert_array_equal(timed_mask, flat_mask)
        np.testing.assert_array_equal(timed_mask, np.asarray(expected_timed_mask))
        assert timed_values.shape == (5292,)

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

    context_values, context_mask = encode_wire_observation(message, expander_context_prior_hinted=True)
    expected_context, expected_context_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, expander_hint=True, context_features=True,
    )
    np.testing.assert_array_equal(context_values, np.asarray(expected_context))
    np.testing.assert_array_equal(context_mask, np.asarray(expected_context_mask))
    assert context_values.shape == (14 * 21 * 21,)
    np.testing.assert_array_equal(context_values.reshape(14, 21, 21)[8], np.asarray(expected.generals))
    np.testing.assert_array_equal(context_values.reshape(14, 21, 21)[9], np.asarray(expected.castles))

    packed_values, packed_mask = encode_wire_observation(message, expander_packed_context_prior_hinted=True)
    expected_packed_context, expected_packed_context_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, expander_hint=True, packed_context_features=True,
    )
    np.testing.assert_array_equal(packed_values, np.asarray(expected_packed_context))
    np.testing.assert_array_equal(packed_mask, np.asarray(expected_packed_context_mask))
    assert packed_values.shape == (10 * 21 * 21,)
    packed_planes = packed_values.reshape(10, 21, 21)
    np.testing.assert_array_equal(
        packed_planes[8], 2 * np.asarray(expected.generals) + np.asarray(expected.castles)
    )

    threat_values, threat_mask = encode_wire_observation(message, expander_neighbor_threat_prior_hinted=True)
    expected_threat, expected_threat_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, expander_hint=True, neighbor_threat_features=True,
    )
    np.testing.assert_array_equal(threat_values, np.asarray(expected_threat))
    np.testing.assert_array_equal(threat_mask, np.asarray(expected_threat_mask))
    assert threat_values.shape == (10 * 21 * 21,)

    distance_values, distance_mask = encode_wire_observation(
        message, expander_general_distance_prior_hinted=True,
    )
    expected_distance, expected_distance_mask = encode_coworld_hinted_observation(
        expected, signed_flags=True, expander_hint=True, general_distance_features=True,
    )
    np.testing.assert_array_equal(distance_values, np.asarray(expected_distance))
    np.testing.assert_array_equal(distance_mask, np.asarray(expected_distance_mask))
    assert distance_values.shape == (10 * 21 * 21,)


def test_neighbor_threat_plane_reaches_adjacent_source_only():
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 2], armies[2, 3] = 10, 9
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, 2] = True
    enemy = np.zeros_like(owned)
    enemy[2, 3] = True
    empty = np.zeros_like(owned)
    observation = Observation(
        armies=jnp.asarray(armies), generals=jnp.asarray(owned), castles=jnp.asarray(empty),
        mountains=jnp.asarray(empty), neutral_cells=jnp.asarray(~(owned | enemy)),
        owned_cells=jnp.asarray(owned), opponent_cells=jnp.asarray(enemy),
        fog_cells=jnp.asarray(empty), structures_in_fog=jnp.asarray(empty),
        owned_land_count=jnp.int32(1), owned_army_count=jnp.int32(10),
        opponent_land_count=jnp.int32(1), opponent_army_count=jnp.int32(9), timestep=jnp.int32(200),
    )
    values, _ = encode_coworld_hinted_observation(
        observation, signed_flags=True, expander_hint=True, neighbor_threat_features=True,
    )
    planes = np.asarray(values).reshape(10, 5, 5)
    assert planes[8, 2, 2] == 2
    assert planes[9, 2, 2] > 1
    assert planes[9, 2, 1] == 0


def test_general_distance_plane_has_zero_at_owned_general():
    armies = np.zeros((5, 5), dtype=np.int32)
    armies[2, 2] = 10
    owned = np.zeros((5, 5), dtype=bool)
    owned[2, 2] = True
    empty = np.zeros_like(owned)
    observation = Observation(
        armies=jnp.asarray(armies), generals=jnp.asarray(owned), castles=jnp.asarray(empty),
        mountains=jnp.asarray(empty), neutral_cells=jnp.asarray(~owned),
        owned_cells=jnp.asarray(owned), opponent_cells=jnp.asarray(empty),
        fog_cells=jnp.asarray(empty), structures_in_fog=jnp.asarray(empty),
        owned_land_count=jnp.int32(1), owned_army_count=jnp.int32(10),
        opponent_land_count=jnp.int32(0), opponent_army_count=jnp.int32(0), timestep=jnp.int32(200),
    )
    values, _ = encode_coworld_hinted_observation(
        observation, signed_flags=True, expander_hint=True, general_distance_features=True,
    )
    distances = np.asarray(values).reshape(10, 5, 5)[9]
    assert distances[2, 2] == 0
    assert distances[2, 3] == distances[3, 2] == 0.1
    assert distances[0, 0] == 0.4


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
    expanded = np.pad(planes, ((0, 0), (0, 6), (0, 0)))
    np.testing.assert_array_equal(
        hinted_replay_indices(expanded.reshape(2, -1), 21, 14),
        np.asarray([[1764, -1], [441 + 123, 1]], dtype=np.int32),
    )
    packed = np.pad(planes, ((0, 0), (0, 2), (0, 0)))
    np.testing.assert_array_equal(
        hinted_replay_indices(packed.reshape(2, -1), 21, 10),
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


@pytest.mark.parametrize('factorized', [True, False])
def test_hosted_action_decode_matches_training_full_half_and_pass(factorized):
    from integrations.puffer_codec import decode_action
    from integrations.softmax.neural_codec import decode_policy_action

    for cell in (0, 220, 440):
        for direction in range(4):
            for split in (0, 1):
                source = direction * 441 + cell
                probabilities = np.zeros(1767 if factorized else 3529, np.float32)
                if factorized:
                    probabilities[source] = 1
                    probabilities[1765 + split] = 1
                    expected = decode_action(source, 21, split)
                else:
                    index = source + 1764 * split
                    probabilities[index] = 1
                    expected = decode_action(index, 21)
                np.testing.assert_array_equal(
                    decode_policy_action(probabilities, factorized_actions=factorized), np.asarray(expected),
                )
    probabilities = np.zeros(1767 if factorized else 3529, np.float32)
    probabilities[1764 if factorized else 3528] = 1
    np.testing.assert_array_equal(
        decode_policy_action(probabilities, factorized_actions=factorized), [1, 0, 0, 0, 0],
    )


def test_hosted_flat_action_sampling_decodes_categorical_choice():
    from integrations.softmax.neural_codec import decode_policy_action

    probabilities = np.zeros(3529, np.float64)
    probabilities[1764] = 1
    assert decode_policy_action(probabilities, factorized_actions=False,
                                rng=np.random.default_rng(7)) == [0, 0, 0, 0, 1]
    probabilities[1764] = 0
    probabilities[3528] = 1
    assert decode_policy_action(probabilities, factorized_actions=False,
                                rng=np.random.default_rng(7)) == [1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="nonnegative flat-action"):
        decode_policy_action(np.ones(1767), factorized_actions=True,
                             rng=np.random.default_rng(7))

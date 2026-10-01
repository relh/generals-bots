"""Convert the public Coworld wire view to the padded Puffer training view."""

import jax
import numpy as np

from generals.core.observation import Observation
from integrations.puffer_codec import (
    calibrate_hint_features,
    encode_coworld_directional_observation,
    encode_coworld_hinted_observation,
    encode_coworld_lean_observation,
    encode_coworld_observation,
    encode_coworld_packed_directional_observation,
    encode_observation,
)


BOARD_SIZE = 21


def decode_policy_action(probabilities, *, factorized_actions: bool = True, rng=None) -> list[int]:
    """Convert either trained action layout to the Coworld move tuple."""
    probabilities = np.asarray(probabilities)
    expected_size = 1767 if factorized_actions else 3529
    if probabilities.shape != (expected_size,) or not np.isfinite(probabilities).all():
        raise ValueError("Policy probabilities differ from the Coworld action layout")
    if rng is not None:
        if factorized_actions or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("Sampling requires nonnegative flat-action probabilities")
        index = int(rng.choice(expected_size, p=probabilities / probabilities.sum()))
        if index == 3528:
            return [1, 0, 0, 0, 0]
        split, source = divmod(index, 1764)
        direction, cell = divmod(source, 441)
        row, col = divmod(cell, 21)
        return [0, row, col, direction, split]
    if factorized_actions:
        source = int(np.argmax(probabilities[:1765]))
        split = int(np.argmax(probabilities[1765:]))
        if source == 1764:
            return [1, 0, 0, 0, 0]
    else:
        index = int(np.argmax(probabilities))
        if index == 3528:
            return [1, 0, 0, 0, 0]
        split, source = divmod(index, 1764)
    direction, cell = divmod(source, 441)
    row, col = divmod(cell, 21)
    return [0, row, col, direction, split]


_calibrate_hints = jax.jit(
    calibrate_hint_features,
    static_argnames=("board_size", "move_hint_scale", "split_hint_scale"),
)
_encode_lean = jax.jit(encode_coworld_lean_observation)
_encode_directional = jax.jit(
    encode_coworld_directional_observation,
    static_argnames=("factorized_actions", "include_timestep", "public_scalar_features", "public_scalar_ablation"),
)
_encode_packed_directional = jax.jit(encode_coworld_packed_directional_observation)
_encode_hinted = jax.jit(encode_coworld_hinted_observation)
_encode_prior_hinted = jax.jit(lambda obs: encode_coworld_hinted_observation(obs, signed_flags=True))
_encode_sprint_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(obs, signed_flags=True, sprint_hint=True)
)
_encode_expander_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(obs, signed_flags=True, expander_hint=True)
)
_encode_expander_context_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(
        obs, signed_flags=True, expander_hint=True, context_features=True,
    )
)
_encode_expander_packed_context_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(
        obs, signed_flags=True, expander_hint=True, packed_context_features=True,
    )
)
_encode_expander_neighbor_threat_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(
        obs, signed_flags=True, expander_hint=True, neighbor_threat_features=True,
    )
)
_encode_expander_general_distance_prior_hinted = jax.jit(
    lambda obs: encode_coworld_hinted_observation(
        obs, signed_flags=True, expander_hint=True, general_distance_features=True,
    )
)


def training_observation(message: dict) -> Observation:
    height, width = message["height"], message["width"]
    if not (18 <= height <= BOARD_SIZE and 18 <= width <= BOARD_SIZE):
        raise ValueError("Expected a Coworld Classic 1v1 board")
    kinds = np.asarray(message["type_grid"], dtype=np.int32)
    owners = np.asarray(message["owner_grid"], dtype=np.int32)
    armies = np.asarray(message["army_grid"], dtype=np.int32)
    if any(grid.shape != (height, width) for grid in (kinds, owners, armies)):
        raise ValueError("Coworld observation grid dimensions differ")

    kind = np.full((BOARD_SIZE, BOARD_SIZE), 5, dtype=np.int32)
    owner = np.zeros_like(kind)
    army = np.zeros_like(kind)
    kind[:height, :width] = kinds
    owner[:height, :width] = owners
    army[:height, :width] = armies
    outside = np.ones_like(kind, dtype=bool)
    outside[:height, :width] = False

    owned = owner == 1
    padded = np.pad(owned, 1)
    visible = np.zeros_like(owned)
    for row in range(3):
        for col in range(3):
            visible |= padded[row : row + BOARD_SIZE, col : col + BOARD_SIZE]

    return Observation(
        armies=army,
        generals=kind == 4,
        castles=kind == 3,
        mountains=(kind == 2) | (outside & visible),
        neutral_cells=(owner == 0) & ((kind == 1) | (kind == 3)),
        owned_cells=owned,
        opponent_cells=owner == 2,
        fog_cells=kind == 0,
        structures_in_fog=((kind == 5) & ~outside) | (outside & ~visible),
        owned_land_count=np.int32(message["my_land"]),
        owned_army_count=np.int32(message["my_army"]),
        opponent_land_count=np.int32(message["opp_land"]),
        opponent_army_count=np.int32(message["opp_army"]),
        timestep=np.int32(message["turn"]),
    )


def encode_wire_observation(
    message: dict, *, compact: bool = False, lean: bool = False,
    directional: bool = False, packed_directional: bool = False, hinted: bool = False,
    prior_hinted: bool = False,
    sprint_prior_hinted: bool = False,
    expander_prior_hinted: bool = False,
    expander_context_prior_hinted: bool = False,
    expander_packed_context_prior_hinted: bool = False,
    expander_neighbor_threat_prior_hinted: bool = False,
    expander_general_distance_prior_hinted: bool = False,
    move_hint_scale: float = 1.0,
    split_hint_scale: float = 1.0,
    factorized_actions: bool = True,
    directional_time_features: bool = False,
    public_scalar_features: bool = False,
    public_scalar_ablation: bool = False,
):
    if not factorized_actions and not directional:
        raise ValueError("Flat Coworld serving requires the directional codec")
    if directional_time_features and not directional:
        raise ValueError("Directional time features require the directional codec")
    if public_scalar_features and not directional:
        raise ValueError("Public scalar features require the directional codec")
    if public_scalar_ablation and not public_scalar_features:
        raise ValueError("Scalar ablation requires public scalar features")
    observation = training_observation(message)
    values, mask = (
        _encode_expander_general_distance_prior_hinted(observation)
        if expander_general_distance_prior_hinted
        else _encode_expander_neighbor_threat_prior_hinted(observation)
        if expander_neighbor_threat_prior_hinted
        else _encode_expander_packed_context_prior_hinted(observation)
        if expander_packed_context_prior_hinted
        else _encode_expander_context_prior_hinted(observation)
        if expander_context_prior_hinted
        else _encode_expander_prior_hinted(observation)
        if expander_prior_hinted
        else _encode_sprint_prior_hinted(observation)
        if sprint_prior_hinted
        else _encode_prior_hinted(observation)
        if prior_hinted
        else _encode_hinted(observation)
        if hinted
        else _encode_packed_directional(observation)
        if packed_directional
        else _encode_directional(
            observation, factorized_actions=factorized_actions, include_timestep=directional_time_features,
            public_scalar_features=public_scalar_features, public_scalar_ablation=public_scalar_ablation,
        )
        if directional
        else _encode_lean(observation)
        if lean
        else encode_coworld_observation(observation)
        if compact
        else encode_observation(observation, factorized_actions=True, goal_features=True)
    )
    if move_hint_scale != 1 or split_hint_scale != 1:
        if not any(
            (
                prior_hinted,
                sprint_prior_hinted,
                expander_prior_hinted,
                expander_context_prior_hinted,
                expander_packed_context_prior_hinted,
                expander_neighbor_threat_prior_hinted,
                expander_general_distance_prior_hinted,
            )
        ):
            raise ValueError("Hint calibration requires signed prior features")
        values = _calibrate_hints(values, 21, move_hint_scale=move_hint_scale, split_hint_scale=split_hint_scale)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=bool)

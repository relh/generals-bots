"""Convert the public Coworld wire view to the padded Puffer training view."""

import numpy as np

from generals.core.observation import Observation
from integrations.puffer_codec import encode_coworld_observation, encode_observation


BOARD_SIZE = 21


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


def encode_wire_observation(message: dict, *, compact: bool = False):
    observation = training_observation(message)
    values, mask = (
        encode_coworld_observation(observation)
        if compact
        else encode_observation(observation, factorized_actions=True, goal_features=True)
    )
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=bool)

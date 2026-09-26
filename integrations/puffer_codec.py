"""Public Generals observation and action contract for native Puffer policies."""

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.harvester_agent import (
    expander_harvester_action, harvester_action, sprint_harvester_action,
)
from generals.agents.hunter_agent import _bfs, _toward
from generals.core.action import compute_valid_move_mask_obs


def harvester_route_features(obs):
    """Public goal and route fields; no teacher action or hidden map state."""
    armies, mine = obs.armies, obs.owned_cells
    size = armies.size
    biggest = jnp.max(jnp.where(mine, armies, 0))
    affordable = obs.castles & obs.neutral_cells & (biggest - 1 > armies)
    passable = ~(obs.mountains | obs.structures_in_fog | (obs.castles & ~mine & ~affordable))
    general = mine & obs.generals
    from_general = _bfs(passable, general)
    enemy_general = obs.opponent_cells & obs.generals
    enemy = obs.opponent_cells & ~obs.castles
    fog = obs.fog_cells & passable & (from_general < size)
    open_land = passable & ~mine & (from_general < size)
    farthest = lambda cells: cells & (from_general == jnp.max(jnp.where(cells, from_general, -1)))
    goal = jnp.where(
        jnp.any(enemy_general), enemy_general,
        jnp.where(jnp.any(affordable), affordable,
                  jnp.where(jnp.any(enemy), enemy,
                            jnp.where(jnp.any(fog), farthest(fog), farthest(open_land)))),
    )
    to_goal = _bfs(passable, goal)
    direction, neighbor = _toward(to_goal, passable)
    advances = neighbor < to_goal
    route = jnp.stack([(direction == i) & advances for i in range(4)])
    return jnp.concatenate((
        jnp.stack((jnp.minimum(from_general, size) / size,
                   jnp.minimum(to_goal, size) / size,
                   goal.astype(jnp.float32))),
        route.astype(jnp.float32),
    ))


def encode_observation(obs, *, factorized_actions=False, goal_features=False):
    planes = obs.as_tensor().astype(jnp.float32)
    for channel in (0, 9, 10, 11, 12):
        planes = planes.at[channel].set(jnp.log1p(jnp.maximum(planes[channel], 0)) / 8.0)
    planes = planes.at[13].set(jnp.minimum(planes[13] / 1200.0, 2.0))
    if goal_features:
        planes = jnp.concatenate((planes, harvester_route_features(obs)))
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    if factorized_actions:
        mask = jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))
    else:
        mask = jnp.concatenate((moves, moves, jnp.ones((1,), dtype=bool)))
    return planes.reshape(-1), mask


def encode_coworld_observation(obs):
    """Compact public 21×21 Classic view with Harvester's route direction."""
    route = harvester_route_features(obs)
    height, width = obs.armies.shape
    land_margin = (obs.owned_land_count - obs.opponent_land_count) / (
        obs.owned_land_count + obs.opponent_land_count + 1
    )
    planes = jnp.stack((
        jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
        obs.generals,
        obs.castles,
        obs.mountains,
        obs.owned_cells,
        obs.opponent_cells,
        obs.fog_cells,
        obs.structures_in_fog,
        jnp.full((height, width), jnp.log1p(obs.owned_army_count) / 8.0),
        jnp.full((height, width), jnp.log1p(obs.opponent_army_count) / 8.0),
        jnp.full((height, width), land_margin),
        jnp.full((height, width), obs.timestep / 1200.0),
        route[1],
        route[4] - route[3] + 2.0 * (route[6] - route[5]),
    )).astype(jnp.float32)
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))


def encode_coworld_lean_observation(obs):
    """Eight public planes for a lower-bandwidth Classic training probe."""
    route = harvester_route_features(obs)
    planes = jnp.stack((
        jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
        obs.generals,
        obs.castles,
        obs.mountains | obs.structures_in_fog,
        obs.owned_cells,
        obs.opponent_cells,
        obs.fog_cells,
        route[4] - route[3] + 2.0 * (route[6] - route[5]),
    )).astype(jnp.float32)
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))


def encode_coworld_directional_observation(obs):
    """Lean public view with one channel for each Harvester route direction."""
    route = harvester_route_features(obs)
    planes = jnp.stack((
        jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
        obs.generals,
        obs.castles,
        obs.mountains | obs.structures_in_fog,
        obs.owned_cells,
        obs.opponent_cells,
        obs.fog_cells,
        route[3], route[4], route[5], route[6],
    )).astype(jnp.float32)
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))


def encode_coworld_packed_directional_observation(obs):
    """Eight-channel Classic view with explicit route and source-selection cues."""
    route = harvester_route_features(obs)
    planes = jnp.stack((
        jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
        obs.owned_cells,
        obs.opponent_cells,
        obs.generals,
        route[3], route[4], route[5], route[6],
    )).astype(jnp.float32)
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))


def encode_coworld_hinted_observation(
    obs, *, signed_flags=False, sprint_hint=False, expander_hint=False,
    context_features=False, packed_context_features=False, neighbor_threat_features=False,
    general_distance_features=False,
):
    """Public view with a scripted move hint for a neural residual policy."""
    hint_fn = expander_harvester_action if expander_hint else sprint_harvester_action if sprint_hint else harvester_action
    hint = hint_fn(jax.random.PRNGKey(0), obs)
    height, width = obs.armies.shape
    direction = jnp.zeros((4, height, width), dtype=jnp.float32)
    direction = direction.at[hint[3], hint[1], hint[2]].set((hint[0] == 0).astype(jnp.float32))
    planes = jnp.concatenate((
        jnp.stack((
            jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
            obs.owned_cells.astype(jnp.float32),
            jnp.full((height, width), 2 * hint[4] - 1 if signed_flags else hint[4], dtype=jnp.float32),
            jnp.full((height, width), 2 * hint[0] - 1 if signed_flags else hint[0], dtype=jnp.float32),
        )),
        direction,
    )).astype(jnp.float32)
    if general_distance_features:
        height, width = obs.generals.shape
        general_cell = jnp.argmax(jnp.ravel(obs.generals & obs.owned_cells))
        rows = jnp.arange(height)[:, None]
        columns = jnp.arange(width)[None, :]
        distance = (jnp.abs(rows - general_cell // width)
                    + jnp.abs(columns - general_cell % width)) / (height + width)
        context = jnp.stack((2.0 * obs.generals + obs.castles, distance)).astype(jnp.float32)
        planes = jnp.concatenate((planes, context))
    elif neighbor_threat_features:
        enemy_strength = jnp.where(
            obs.opponent_cells,
            1.0 + jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
            0.0,
        )
        nearby_enemy = jax.lax.reduce_window(
            enemy_strength, 0.0, jax.lax.max, (3, 3), (1, 1), "SAME"
        )
        context = jnp.stack((2.0 * obs.generals + obs.castles, nearby_enemy)).astype(jnp.float32)
        planes = jnp.concatenate((planes, context))
    elif packed_context_features:
        context = jnp.stack((
            2.0 * obs.generals + obs.castles,
            obs.opponent_cells.astype(jnp.float32) - (obs.mountains | obs.structures_in_fog),
        )).astype(jnp.float32)
        planes = jnp.concatenate((planes, context))
    elif context_features:
        context = jnp.stack((
            obs.generals,
            obs.castles,
            obs.opponent_cells,
            obs.mountains | obs.structures_in_fog,
            jnp.full((height, width), jnp.log1p(obs.owned_army_count) / 8.0),
            jnp.full((height, width), jnp.log1p(obs.opponent_army_count) / 8.0),
        )).astype(jnp.float32)
        planes = jnp.concatenate((planes, context))
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))


def hinted_replay_indices(values, board_size: int, channels: int = 8):
    """Read the exact scripted action already present in signed hint planes."""
    cells = board_size * board_size
    planes = np.asarray(values).reshape(-1, channels, cells)
    source = np.argmax(planes[:, 4:8].reshape(-1, 4 * cells), axis=1).astype(np.int32)
    passing = planes[:, 3, 0] > 0
    split = np.where(passing, -1, planes[:, 2, 0] > 0).astype(np.int32)
    return np.stack((np.where(passing, 4 * cells, source), split), axis=1)


def hinted_teacher_action_device(values, board_size: int):
    """Recover the deterministic teacher action already encoded in public hint planes."""
    cells = board_size * board_size
    planes = values.reshape(-1, cells)
    source = jnp.argmax(planes[4:8].reshape(-1))
    direction, cell = jnp.divmod(source, cells)
    passing = planes[3, 0] > 0
    split = planes[2, 0] > 0
    return jnp.array((passing, cell // board_size, cell % board_size, direction, split), dtype=jnp.int32)


def decode_action(index, size, split=None):
    cells = size * size
    channel, position = index // cells, index % cells
    if split is None:
        split = channel // 4
        pass_index = 8 * cells
    else:
        pass_index = 4 * cells
    move = jnp.array([0, position // size, position % size, channel % 4, split], dtype=jnp.int32)
    return jnp.where(index == pass_index, jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32), move)

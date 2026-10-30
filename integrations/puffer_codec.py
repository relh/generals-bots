"""Public Generals observation and action contract for native Puffer policies."""

import jax.numpy as jnp

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

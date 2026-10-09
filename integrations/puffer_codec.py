"""Canonical 16-plane, 3529-action padded Classic policy contract."""

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
    def farthest(cells):
        return cells & (from_general == jnp.max(jnp.where(cells, from_general, -1)))
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


def encode_coworld_directional_observation(obs):
    """Encode public Classic maps, routes and five broadcast economic scalars."""
    if obs.armies.shape != (21, 21):
        raise ValueError("Classic policy observations must be padded to 21×21")
    route = harvester_route_features(obs)
    planes = jnp.stack((
        jnp.log1p(jnp.maximum(obs.armies, 0)) / 8.0,
        obs.generals, obs.castles, obs.mountains | obs.structures_in_fog,
        obs.owned_cells, obs.opponent_cells, obs.fog_cells,
        route[3], route[4], route[5], route[6],
    )).astype(jnp.float32)
    scalars = jnp.asarray((
        obs.timestep / 2000.0, obs.owned_land_count / 441.0,
        obs.opponent_land_count / 441.0,
        jnp.log1p(jnp.maximum(obs.owned_army_count, 0)) / 8.0,
        jnp.log1p(jnp.maximum(obs.opponent_army_count, 0)) / 8.0,
    ), dtype=jnp.float32)
    planes = jnp.concatenate((planes, jnp.broadcast_to(scalars[:, None, None], (5, 21, 21))))
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    return planes.reshape(-1), jnp.concatenate((moves, moves, jnp.ones((1,), dtype=bool)))


def decode_action(index, size=21):
    """Decode four full and four half move planes, followed by pass."""
    if size != 21:
        raise ValueError("Classic policy actions use a padded 21×21 board")
    channel, position = index // 441, index % 441
    move = jnp.array([0, position // 21, position % 21, channel % 4, channel // 4], dtype=jnp.int32)
    return jnp.where(index == 3528, jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32), move)

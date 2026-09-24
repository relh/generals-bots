"""Harvester: Hunter, plus it banks the cities Hunter walks past.

A minimal improvement on the Hunter. Army income is dominated by *structures*:
the general and every owned **city** grow +1 every 2 turns, while plain land
grows only +1 per 50 turns. Hunter and Expander both route around cities and
never take them. The Harvester runs Hunter's exact garrison / conveyor / kill
machinery, but once its stack is big enough to afford a neutral castle it detours
to seize it first — every city is a permanent +0.5 army/turn engine, so it
out-masses a city-less Hunter and then decapitates it the same way.
"""
import jax
import jax.numpy as jnp

from generals.core.observation import Observation
from generals.core.action import compute_valid_move_mask_obs

from .agent import Agent
from .hunter_agent import GARRISON, _bfs, _toward


@jax.jit
def _harvester_action(key, obs, fast_opening, fortify_general=False):
    """Capture general > seize an affordable city > feed surplus > advance > wait."""
    del key
    a, mine = obs.armies, obs.owned_cells
    H, W = a.shape
    reach = jnp.int32(H * W)
    mine_army = jnp.where(mine, a, 0)
    movable = mine & (a > 1)
    biggest = jnp.max(mine_army)

    gen = mine & obs.generals
    gen_army = jnp.sum(jnp.where(gen, a, 0))
    g = jnp.argmax(gen.reshape(-1).astype(jnp.int32))

    # Affordable neutral castles become walkable targets; other cities stay walls.
    affordable_city = obs.cities & obs.neutral_cells & (biggest - 1 > a)
    passable = ~(obs.mountains | obs.structures_in_fog | (obs.cities & ~mine & ~affordable_city))
    from_gen = _bfs(passable, gen)

    # Goal: enemy general > an affordable city to bank > nearest enemy land > scout.
    egen = obs.opponent_cells & obs.generals
    enemy = obs.opponent_cells & ~obs.cities
    fog = obs.fog_cells & passable & (from_gen < reach)
    open_ = passable & ~mine & (from_gen < reach)
    farthest = lambda m: m & (from_gen == jnp.max(jnp.where(m, from_gen, -1)))
    goal = jnp.where(jnp.any(egen), egen,
           jnp.where(jnp.any(affordable_city), affordable_city,
           jnp.where(jnp.any(enemy), enemy,
           jnp.where(jnp.any(fog), farthest(fog), farthest(open_)))))

    to_goal = _bfs(passable, goal)
    direction, nbr = _toward(to_goal, passable)
    advances = nbr < to_goal
    dirn = direction.reshape(-1)

    egen_army = jnp.sum(jnp.where(egen, a, 0))
    kill = jnp.any(egen) & movable & (to_goal == 1) & advances & (a - 1 > egen_army)
    ki = jnp.argmax(jnp.where(kill, mine_army, -1).reshape(-1))
    sprinting = fast_opening & (obs.timestep < 80)
    reserve = fortify_general & (obs.timestep >= 100)
    feed_floor = jnp.where(reserve, 120, jnp.where(sprinting, 2, 2 * GARRISON))
    feed = (gen_army >= feed_floor) & advances.reshape(-1)[g]
    fwd = movable & ~gen & advances
    ci = jnp.argmax(jnp.where(fwd, mine_army, -1).reshape(-1))

    do_kill = jnp.any(kill)
    do_feed = ~do_kill & feed
    do_conv = ~do_kill & ~do_feed & jnp.any(fwd)
    i = jnp.where(do_kill, ki, jnp.where(do_feed, g, ci))
    return jnp.array([~(do_kill | do_feed | do_conv), i // W, i % W, dirn[i],
                      do_feed & ~sprinting], dtype=jnp.int32)


@jax.jit
def harvester_action(key, obs):
    return _harvester_action(key, obs, False)


@jax.jit
def sprint_harvester_action(key, obs):
    """Expand as soon as the general can move, then preserve the usual garrison."""
    return _harvester_action(key, obs, True)


@jax.jit
def fortified_harvester_action(key, obs):
    """Keep a growing home garrison after the opening while scouting outward."""
    return _harvester_action(key, obs, True, True)


class HarvesterAgent(Agent):
    """Runs Hunter's playbook but banks affordable cities for a runaway economy."""

    def __init__(self, id: str = "Harvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return harvester_action(key, observation)

    def reset(self):
        pass


class SprintHarvesterAgent(HarvesterAgent):
    def __init__(self, id: str = "SprintHarvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return sprint_harvester_action(key, observation)


def _owned_six_hop_distances(owned, root):
    """Bound a castle rally to the same six owned steps as the incumbent."""
    height, width = owned.shape
    far = jnp.int32(height * width + 5)
    distance = jnp.where(root, jnp.int32(0), far)

    def relax(_, field):
        neighbors = jnp.minimum(
            jnp.minimum(jnp.roll(field, 1, 0).at[0].set(far),
                        jnp.roll(field, -1, 0).at[-1].set(far)),
            jnp.minimum(jnp.roll(field, 1, 1).at[:, 0].set(far),
                        jnp.roll(field, -1, 1).at[:, -1].set(far)),
        )
        return jnp.where(owned, jnp.minimum(field, neighbors + 1), far)

    return jax.lax.fori_loop(0, 6, relax, distance)


@jax.jit
def expander_harvester_action(key, obs):
    """Expand, rally for castles, and reinforce home when a visible army threatens it."""
    height, width = obs.armies.shape
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
    toward_army = jnp.stack([jnp.roll(obs.armies, shift, (0, 1)) for shift in shifts])
    toward_owned = jnp.stack([jnp.roll(obs.owned_cells, shift, (0, 1)) for shift in shifts])
    toward_enemy = jnp.stack([jnp.roll(obs.opponent_cells, shift, (0, 1)) for shift in shifts])
    toward_castle = jnp.stack([jnp.roll(obs.castles, shift, (0, 1)) for shift in shifts])
    toward_neutral = jnp.stack([jnp.roll(obs.neutral_cells, shift, (0, 1)) for shift in shifts])
    toward_general = jnp.stack([jnp.roll(obs.generals, shift, (0, 1)) for shift in shifts])
    toward_blocked = jnp.stack([jnp.roll(obs.structures_in_fog, shift, (0, 1)) for shift in shifts])
    legal = compute_valid_move_mask_obs(obs).transpose(2, 0, 1)
    home = obs.generals & obs.owned_cells
    home_cell = jnp.argmax(home.reshape(-1).astype(jnp.int32))
    home_row, home_col = jnp.divmod(home_cell, width)
    rr = jnp.broadcast_to(jnp.arange(height)[:, None], (height, width))
    cc = jnp.broadcast_to(jnp.arange(width)[None, :], (height, width))
    close_enemy = (obs.opponent_cells & (jnp.maximum(jnp.abs(rr - home_row),
                                                    jnp.abs(cc - home_col)) <= 5))
    threat_army = jnp.max(jnp.where(close_enemy, obs.armies, 0))
    threatened = (obs.timestep >= 100) & (threat_army >= 8)
    protected_general = home[None] & threatened
    general_target = toward_enemy & toward_general
    captures = (legal & ~toward_owned & ~toward_blocked
                & (obs.armies[None] > toward_army + 1)
                & (~protected_general | general_target))
    priority = (3 * (toward_enemy & toward_general).astype(jnp.int32)
                + 2 * toward_castle.astype(jnp.int32)
                + toward_enemy.astype(jnp.int32))
    source_rank = jnp.where((obs.timestep >= 800) & toward_enemy,
                            obs.armies[None], -obs.armies[None])
    row = jnp.arange(height)[None, :, None]
    col = jnp.arange(width)[None, None, :]
    edge = jnp.minimum(jnp.minimum(row, height - 1 - row),
                       jnp.minimum(col, width - 1 - col))
    score = priority * 1000000 + source_rank * 1000 - toward_army * 10 + edge
    index = jnp.argmax(jnp.where(captures, score, -1000000000).reshape(-1))
    direction, cell = jnp.divmod(index, height * width)
    action = jnp.array([0, cell // width, cell % width, direction, 0], dtype=jnp.int32)

    # A cheap land capture must not preempt a reachable city that the connected
    # owned component can afford. Pick its staging tile, then transfer surplus
    # along owned cells until that tile can take the defenders.
    in_bounds = jnp.stack((rr > 0, rr < height - 1, cc > 0, cc < width - 1))
    city_roots = (obs.owned_cells[None] & ~protected_general & toward_castle & toward_neutral
                  & ~toward_blocked & in_bounds)
    city_rank = -toward_army * 1000 + obs.armies[None]
    city_index = jnp.argmax(jnp.where(city_roots, city_rank, -1000000000).reshape(-1))
    city_direction, root_cell = jnp.divmod(city_index, height * width)
    root = jnp.arange(height * width).reshape(height, width) == root_cell
    owned_distance = _owned_six_hop_distances(obs.owned_cells, root)
    connected = obs.owned_cells & (owned_distance < height * width)
    surplus = jnp.sum(jnp.where(connected, jnp.maximum(obs.armies - 1, 0), 0))
    defense = toward_army.reshape(-1)[city_index]
    root_army = obs.armies.reshape(-1)[root_cell]
    gather_direction, neighbor_distance = _toward(owned_distance, obs.owned_cells)
    feeders = (connected & (owned_distance > 0) & (obs.armies > 1)
               & ~protected_general[0] & (neighbor_distance < owned_distance))
    feeder = jnp.argmax(jnp.where(feeders, obs.armies * 1000 - owned_distance, -1).reshape(-1))
    can_take_city = root_army > defense + 1
    city_action = jnp.array([0, root_cell // width, root_cell % width, city_direction, 0], dtype=jnp.int32)
    gather_action = jnp.array([0, feeder // width, feeder % width,
                               gather_direction.reshape(-1)[feeder], 0], dtype=jnp.int32)
    rally_action = jnp.where(can_take_city, city_action, gather_action)
    can_rally = (jnp.any(city_roots) & (surplus > defense + 2)
                 & (can_take_city | jnp.any(feeders)))

    # A nearby enemy stack can reach the crown before passive income catches
    # up. Transfer the strongest connected surplus toward home when its reserve
    # falls short, while leaving normal expansion untouched without a threat.
    home_distance = _owned_six_hop_distances(obs.owned_cells, home)
    home_direction, home_neighbor_distance = _toward(home_distance, obs.owned_cells)
    home_feeders = (obs.owned_cells & ~home & (home_distance > 0)
                    & (home_distance < height * width) & (obs.armies > 1)
                    & (home_neighbor_distance < home_distance))
    home_feeder = jnp.argmax(jnp.where(
        home_feeders, obs.armies * 1000 - home_distance, -1
    ).reshape(-1))
    home_action = jnp.array([0, home_feeder // width, home_feeder % width,
                             home_direction.reshape(-1)[home_feeder], 0], dtype=jnp.int32)
    reinforce_home = (threatened & jnp.any(home_feeders)
                      & (obs.armies.reshape(-1)[home_cell] < threat_army + 30))

    visible_general = jnp.any(obs.opponent_cells & obs.generals)
    immediate_general_capture = jnp.any(captures & toward_enemy & toward_general)
    immediate_city_capture = jnp.any(captures & toward_castle)
    route_action = _harvester_action(key, obs, True, threatened)
    return jnp.where(
        immediate_general_capture, action,
        jnp.where(reinforce_home, home_action,
                  jnp.where(visible_general, route_action,
                            jnp.where(immediate_city_capture, action,
                                      jnp.where(can_rally, rally_action,
                                                jnp.where(jnp.any(captures), action, route_action))))),
    )


class ExpanderHarvesterAgent(HarvesterAgent):
    def __init__(self, id: str = "ExpanderHarvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return expander_harvester_action(key, observation)

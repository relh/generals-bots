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
def _harvester_action(key, obs, fast_opening):
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
    feed = (gen_army >= jnp.where(sprinting, 2, 2 * GARRISON)) & advances.reshape(-1)[g]
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


@jax.jit
def expander_harvester_action(key, obs):
    """Take affordable frontier cells first; use the sprint route between captures."""
    height, width = obs.armies.shape
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1))
    toward_army = jnp.stack([jnp.roll(obs.armies, shift, (0, 1)) for shift in shifts])
    toward_owned = jnp.stack([jnp.roll(obs.owned_cells, shift, (0, 1)) for shift in shifts])
    toward_enemy = jnp.stack([jnp.roll(obs.opponent_cells, shift, (0, 1)) for shift in shifts])
    toward_castle = jnp.stack([jnp.roll(obs.castles, shift, (0, 1)) for shift in shifts])
    toward_general = jnp.stack([jnp.roll(obs.generals, shift, (0, 1)) for shift in shifts])
    toward_blocked = jnp.stack([jnp.roll(obs.structures_in_fog, shift, (0, 1)) for shift in shifts])
    legal = compute_valid_move_mask_obs(obs).transpose(2, 0, 1)
    captures = legal & ~toward_owned & ~toward_blocked & (obs.armies[None] > toward_army + 1)
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
    return jax.lax.cond(jnp.any(captures), lambda _: action,
                        lambda _: sprint_harvester_action(key, obs), operand=None)


class ExpanderHarvesterAgent(HarvesterAgent):
    def __init__(self, id: str = "ExpanderHarvester"):
        super().__init__(id)

    def act(self, observation: Observation, key: jnp.ndarray) -> jnp.ndarray:
        return expander_harvester_action(key, observation)

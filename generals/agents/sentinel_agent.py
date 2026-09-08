"""Observation-only tactical commander for classic and competition games.

Sentinel routes surplus armies toward a shared objective, considers full and half
moves, and checks general safety before spending its army. There is no hidden
state access or Python episode memory; the same policy supports jit and vmap.
"""

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs
from generals.core.observation import Observation

from .agent import Agent


def _neighbors(a, fill):
    """Destination values in UP, DOWN, LEFT, RIGHT order, shape (H,W,4)."""
    return jnp.stack(
        (
            jnp.roll(a, 1, 0).at[0].set(fill),
            jnp.roll(a, -1, 0).at[-1].set(fill),
            jnp.roll(a, 1, 1).at[:, 0].set(fill),
            jnp.roll(a, -1, 1).at[:, -1].set(fill),
        ),
        axis=-1,
    )


def _distance(passable, sources, costs=None):
    """Convergent Bellman distances, stopping when no distances change."""
    inf = jnp.float32(1e6)
    costs = jnp.ones(passable.shape) if costs is None else costs
    initial = jnp.where(sources, 0.0, inf)

    def cond(carry):
        step, _, changed = carry
        return (step < passable.size) & changed

    def body(carry):
        step, field, _ = carry
        candidate = jnp.min(_neighbors(field + costs, inf), axis=-1)
        new = jnp.where(sources, 0.0, jnp.where(passable, jnp.minimum(field, candidate), inf))
        return step + 1, new, jnp.any(new != field)

    return jax.lax.while_loop(cond, body, (0, initial, jnp.array(True)))[1]


def _build_prices(structures):
    """Exact observable build prices, matching the modifier's Manhattan kernel."""
    h, w = structures.shape
    padded = jnp.pad(structures.astype(jnp.int32), 6)
    price = jnp.full((h, w), 35, jnp.int32)
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            surcharge = 14 - 2 * (abs(dr) + abs(dc))
            if surcharge > 0:
                price = price + surcharge * padded[6 + dr : 6 + dr + h, 6 + dc : 6 + dc + w]
    return price


class SentinelAgent(Agent):
    """Strategic baseline with explicit rules and inspectable decision telemetry.

    ``decision(obs, key)`` returns (action, telemetry); ``act`` returns only the
    standard five integers. Rule settings must match the environment. No build
    action is emitted unless build_castles=True.
    """

    def __init__(self, id="Sentinel", *, build_castles=False, deathtouch_turn=None, max_turns=1200):
        super().__init__(id)
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns

    @partial(jax.jit, static_argnums=0)
    def act(self, observation: Observation, key):
        return self.decision(observation, key)[0]

    @partial(jax.jit, static_argnums=0)
    def decision(self, obs: Observation, key):
        del key  # Stable decisions aid replay diagnosis and paired comparisons.
        a, mine = obs.armies, obs.owned_cells
        h, w = a.shape
        gen = mine & obs.generals
        egen = obs.opponent_cells & obs.generals
        own_structures = mine & (obs.generals | obs.castles)
        gen_army = jnp.sum(jnp.where(gen, a, 0))
        biggest = jnp.max(jnp.where(mine, a, 0))
        active_touch = obs.timestep >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        allied = jnp.zeros_like(mine) if obs.allied_cells is None else obs.allied_cells
        friendly = mine | allied
        # Keep expensive neutral castles out of scouting routes. Enemy structures
        # remain potential targets, including an enemy general with any army.
        affordable = obs.castles & obs.neutral_cells & (a + 6 < biggest)
        passable = ~(obs.mountains | obs.structures_in_fog | (obs.castles & obs.neutral_cells & ~affordable))
        home_distance = _distance(passable, gen)
        enemy_force = jnp.where(obs.opponent_cells, jnp.maximum(a - 1, 0), 0)
        local_threat = jnp.max(_neighbors(enemy_force, 0), axis=-1)
        immediate = obs.opponent_cells & (home_distance == 1) & (a > 1)
        imminent_army = jnp.max(jnp.where(immediate, a - 1, 0))
        near_force = jnp.max(jnp.where(obs.opponent_cells & (home_distance <= 3), jnp.maximum(a - home_distance, 0), 0))
        # A distant stack must first fight the outgoing army. Reserving its
        # entire strength at home strands all income in two-step corridors.
        # Discount by distance; adjacent threats retain their full reserve.
        reserve_force = jnp.max(
            jnp.where(
                obs.opponent_cells & (home_distance <= 3),
                jnp.maximum(a - home_distance, 0) / jnp.maximum(home_distance, 1),
                0,
            )
        )
        reserve = jnp.maximum(3.0, reserve_force + 1)
        # A city pays back its army investment over the remaining horizon. Avoid
        # detours while the home general is under attack.
        city_value = jnp.where(affordable & (home_distance < 1e5), 50 - a * 0.4 - home_distance, -1e6)
        city_index = jnp.argmax(city_value)
        city_target = jnp.arange(h * w).reshape(h, w) == city_index
        take_city = jnp.any(affordable) & (near_force < gen_army) & ~jnp.any(egen)
        enemy = obs.opponent_cells & passable
        fog = obs.fog_cells & passable & (home_distance < 1e5)
        open_land = passable & ~friendly & (home_distance < 1e5)
        scout = jnp.where(jnp.any(fog), fog, open_land)
        farthest = scout & (home_distance == jnp.max(jnp.where(scout, home_distance, -1)))
        goal = jnp.where(
            jnp.any(egen), egen, jnp.where(take_city, city_target, jnp.where(jnp.any(enemy), enemy, farthest))
        )
        costs = 1 + jnp.where(~friendly, a, 0) * 0.12
        to_goal = _distance(passable, goal, costs)
        dest_distance = _neighbors(to_goal, 1e6)
        advances = dest_distance < to_goal[..., None]
        dest_a = _neighbors(a, 0)[..., None]
        dest_mine = _neighbors(friendly, False)[..., None]
        dest_enemy = _neighbors(obs.opponent_cells, False)[..., None]
        dest_gen = _neighbors(gen, False)[..., None]
        dest_egen = _neighbors(egen, False)[..., None]
        dest_castle = _neighbors(obs.castles, False)[..., None]
        dest_fog = _neighbors(obs.fog_cells, False)[..., None]
        moved = jnp.stack((a - 1, a // 2), axis=-1)[..., None, :]
        remaining = a[..., None, None] - moved
        captures = ~dest_mine & (moved > dest_a)
        kill = dest_egen & ((moved > dest_a) | active_touch)
        valid = (
            compute_valid_move_mask_obs(obs)[..., None]
            & ~_neighbors(obs.structures_in_fog, True)[..., None]
            & (moved > 0)
        )
        # Suicide attacks waste turns and feed an opponent a stationary target.
        valid &= dest_mine | captures | kill
        surplus = jnp.where(dest_mine, moved + dest_a, moved - dest_a)
        safety = surplus - _neighbors(local_threat, 0)[..., None]
        scores = (
            advances[..., None] * (5 + moved * 0.65)
            + captures * (3 + dest_enemy * 3 + dest_castle * 18)
            + dest_fog * 1.5
            + dest_mine * jnp.minimum(dest_a, moved) * 0.10
            - jnp.where(captures, dest_a * 0.12, 0)
            - jnp.maximum(-safety, 0) * 0.8
        )
        # Friendly transfers must advance the campaign or reinforce the general.
        scores = jnp.where(dest_mine & ~advances[..., None], -5.0, scores)
        # In a close general standoff, one safe founding army can buy the land
        # income that breaks an otherwise permanent equal-growth stalemate.
        # The hard immediate-threat check below still forbids a losing sortie.
        close_standoff = jnp.any(egen & (home_distance == 1)) & (obs.owned_land_count == 1)
        founding = close_standoff & captures & ~dest_enemy & ~dest_castle & (moved == 1)
        reserve_penalty = jnp.where(founding, 0.5, 12.0)
        scores -= gen[..., None, None] * jnp.maximum(reserve - remaining, 0) * reserve_penalty
        # Evaluate the largest still-live adjacent attacker for each candidate.
        # Third-tile interception works even after deathtouch; head-on attacks by
        # our own general are deliberately not credited as safe interception.
        removed_threat = (
            dest_enemy & captures & (_neighbors(home_distance, 1e6)[..., None] == 1) & ~gen[..., None, None]
        )
        second_threat = jnp.sort(jnp.where(immediate, a - 1, 0).reshape(-1))[-2]
        after_threat = jnp.where(removed_threat & (dest_a - 1 >= imminent_army), second_threat, imminent_army)
        after_home = gen_army - gen[..., None, None] * moved + dest_gen * moved
        unsafe = (after_threat > after_home) | (active_touch & (after_threat > 0))
        scores -= unsafe * (2000 + jnp.maximum(after_threat - after_home, 0) * 20)
        # When threatened but not immediately doomed, intercept or move reserves
        # toward home instead of continuing an unrelated expedition.
        toward_home = _neighbors(home_distance, 1e6) < home_distance[..., None]
        emergency = near_force >= gen_army
        scores += emergency * toward_home[..., None] * dest_mine * moved * 2
        scores += removed_threat * (100 + moved)
        scores += kill * 100000
        scores = jnp.where(valid, scores, -1e9)
        flat = jnp.argmax(scores)
        split = flat % 2
        direction = (flat // 2) % 4
        cell = flat // 8
        score = scores.reshape(-1)[flat]
        pass_unsafe = (imminent_army > gen_army) | (active_touch & (imminent_army > 0))
        pass_score = jnp.where(pass_unsafe, -2000 - jnp.maximum(imminent_army - gen_army, 0) * 20, 0)
        action = jnp.array([score <= pass_score, cell // w, cell % w, direction, split], jnp.int32)
        building = jnp.array(False)
        if self.build_castles:
            price = _build_prices(own_structures)
            # Require repayment time plus a useful profit window, and preserve a
            # buffer against visible attacks. Existing structures compound income.
            safe_build = (
                mine
                & ~own_structures
                & (a >= price + 5 + local_threat)
                & (home_distance >= 2)
                & (near_force < gen_army)
                & (obs.timestep + 2 * price + 100 < self.max_turns)
            )
            build_score = jnp.where(safe_build, 45 + (self.max_turns - obs.timestep) * 0.08 - price * 0.3, -1e9)
            bi = jnp.argmax(build_score)
            building = jnp.max(build_score) > jnp.maximum(score, pass_score)
            action = jnp.where(building, jnp.array([2, bi // w, bi % w, 0, 0], jnp.int32), action)
        telemetry = {
            "score": score,
            "general_army": gen_army,
            "general_reserve": reserve,
            "adjacent_threat": imminent_army,
            "goal_visible_general": jnp.any(egen),
            "goal_city": take_city,
            "building": building,
            "deathtouch_active": active_touch,
            "candidate_count": jnp.sum(valid),
        }
        return action, telemetry

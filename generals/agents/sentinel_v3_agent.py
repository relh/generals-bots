"""Experimental stateful defense layered over the unchanged Sentinel v2 policy.

Remembered armies describe possible threats, never fabricated observation cells.
The max envelope avoids adding copies of a single uncertain army together.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs

from .sentinel_agent import SentinelAgent, _build_prices, _distance, _neighbors


# Deliberate frozen-v2 scoring copy: only the extra scores return differs.
# Keep action-parity tests when changing this local extraction; never edit v2.
def _campaign_decision(self, obs, key):
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
    goal = jnp.where(jnp.any(egen), egen, jnp.where(take_city, city_target, jnp.where(jnp.any(enemy), enemy, farthest)))
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
        compute_valid_move_mask_obs(obs)[..., None] & ~_neighbors(obs.structures_in_fog, True)[..., None] & (moved > 0)
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
    removed_threat = dest_enemy & captures & (_neighbors(home_distance, 1e6)[..., None] == 1) & ~gen[..., None, None]
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
    return action, telemetry, scores


class SentinelMemory(NamedTuple):
    threat_army: jax.Array
    threat_age: jax.Array
    last_turn: jax.Array
    reserve: jax.Array
    defend_until: jax.Array


class SentinelV3Agent:
    """Bounded prototype; memory and sustained-defense ablations are independent.

    ``initial_memory(shape)`` starts an episode. ``step(obs,key,memory)`` returns
    action, updated memory, and scalar telemetry. Memory belongs to one player
    in one episode; callers must reset it between episodes.
    """

    MEMORY_TTL = 24
    DEFENSE_HOLD = 10

    def __init__(
        self,
        id="Sentinel-v3",
        *,
        build_castles=False,
        deathtouch_turn=None,
        max_turns=1200,
        remember_threats=True,
        sustained_defense=True,
    ):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.remember_threats = remember_threats
        self.sustained_defense = sustained_defense
        self.baseline = SentinelAgent(
            build_castles=build_castles,
            deathtouch_turn=deathtouch_turn,
            max_turns=max_turns,
        )

    def initial_memory(self, shape):
        return SentinelMemory(
            jnp.zeros(shape, jnp.float32),
            jnp.zeros(shape, jnp.int32),
            jnp.int32(-1),
            jnp.float32(3),
            jnp.int32(-1),
        )

    def _threat_memory(self, obs, memory, passable):
        elapsed = jnp.clip(obs.timestep - memory.last_turn, 0, self.MEMORY_TTL + 1)
        reset = (memory.last_turn < 0) | (obs.timestep < memory.last_turn)
        army = jnp.where(reset, 0, memory.threat_army)
        age = jnp.where(reset, 0, memory.threat_age)

        def advance(_, carry):
            force, old_age = carry
            forces = jnp.concatenate((force[..., None], _neighbors(force, 0)), axis=-1)
            ages = jnp.concatenate((old_age[..., None], _neighbors(old_age, self.MEMORY_TTL)), axis=-1)
            choice = jnp.argmax(forces, axis=-1)[..., None]
            new_age = jnp.take_along_axis(ages, choice, axis=-1)[..., 0] + 1
            new_force = jnp.maximum(jnp.max(forces, axis=-1) - 0.75, 0)
            return jnp.where(passable & (new_age <= self.MEMORY_TTL), new_force, 0), new_age

        army, age = jax.lax.fori_loop(0, elapsed, advance, (army, age))
        # A visible empty/friendly/small-enemy cell disproves that location, but
        # does not disprove other reachable fog locations of the same old army.
        fog = obs.fog_cells & passable
        army = jnp.where(fog & self.remember_threats, army, 0)
        observed = obs.opponent_cells & ~obs.generals & (obs.armies >= 8)
        army = jnp.where(observed, obs.armies.astype(jnp.float32), army)
        age = jnp.where(observed | (army == 0), 0, age)
        # Public global army counts also bound uncertain local estimates.
        army = jnp.minimum(army, obs.opponent_army_count.astype(jnp.float32))
        return army, age, observed

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        base, telemetry, campaign_scores = _campaign_decision(self.baseline, obs, key)
        a, mine = obs.armies, obs.owned_cells
        h, w = a.shape
        general = mine & obs.generals
        home_army = jnp.sum(jnp.where(general, a, 0))
        # Defensive routes do not promise travel through defended neutral cities.
        passable = ~(obs.mountains | obs.structures_in_fog | (obs.castles & obs.neutral_cells))
        distance = _distance(passable, general)
        threats, ages, observed = self._threat_memory(obs, memory, passable)
        # Allow for movement attrition and home growth before earliest arrival.
        # Six spare troops protect against a follow-up wave and rounding.
        arrival = jnp.maximum(threats - 1.5 * distance + 6, 0)
        relevant = (threats >= 8) & (distance <= 12) & (distance > 0)
        visible_reserve = jnp.max(jnp.where(relevant & observed, arrival, 0))
        fog_reserve = jnp.max(jnp.where(relevant & ~observed, arrival, 0))
        requested = jnp.maximum(visible_reserve, fog_reserve)
        credible = requested >= jnp.maximum(6, home_army * 0.6)
        current_reserve = jnp.where(credible, requested, 3.0)
        until = jnp.where(credible, obs.timestep + self.DEFENSE_HOLD, memory.defend_until)
        held = jnp.where(obs.timestep <= memory.defend_until, memory.reserve, jnp.maximum(3, memory.reserve - 2))
        reserve = jnp.maximum(current_reserve, held) if self.sustained_defense else current_reserve
        enabled = self.remember_threats or self.sustained_defense
        reserve = jnp.where(enabled, reserve, 3.0)
        active = enabled & (reserve > 3) & (obs.owned_land_count > 1)

        moved = jnp.stack((a - 1, a // 2), axis=-1)[..., None, :]
        dest_a = _neighbors(a, 0)[..., None]
        dest_mine = _neighbors(mine, False)[..., None]
        dest_general = _neighbors(general, False)[..., None]
        dest_enemy = _neighbors(obs.opponent_cells, False)[..., None]
        toward_home = (_neighbors(distance, 1e6) < distance[..., None])[..., None]
        captures = ~dest_mine & (moved > dest_a)
        valid = (
            compute_valid_move_mask_obs(obs)[..., None]
            & (moved > 0)
            & ~_neighbors(obs.structures_in_fog, True)[..., None]
            & ~general[..., None, None]
            & toward_home
            & (dest_mine | captures)
        )
        imminent = telemetry["adjacent_threat"]
        adjacent_enemy = obs.opponent_cells & (distance == 1)
        second = jnp.sort(jnp.where(adjacent_enemy, a - 1, 0).reshape(-1))[-2]
        intercept = dest_enemy & captures & (_neighbors(distance, 1e6)[..., None] == 1)
        remaining_threat = jnp.where(intercept & (dest_a - 1 >= imminent), second, imminent)
        touch = telemetry["deathtouch_active"]
        safe = (remaining_threat <= home_army + dest_general * moved) & ~(touch & (remaining_threat > 0))
        score = moved / jnp.maximum(distance[..., None, None], 1) + intercept * 100 + dest_general * 10
        score = jnp.where(valid & safe, score, -1e9)
        index = jnp.argmax(score)
        cell = index // 8
        defensive = jnp.array([0, cell // w, cell % w, (index // 2) % 4, index % 2], jnp.int32)
        defensive = jnp.where(jnp.max(score) > 0, defensive, jnp.array([1, 0, 0, 0, 0], jnp.int32))

        _, r, c, direction, split = base
        sent = jnp.where(split == 1, a[r, c] // 2, a[r, c] - 1)
        outgoing = (base[0] == 0) & general[r, c]
        would_deplete = outgoing & (home_army - sent < reserve)
        # The six-troop planning buffer can retain a garrison, but must not
        # withdraw a campaign merely to fill that buffer. Arrival estimates
        # already account for the general's growth before an invasion arrives.
        need_reinforcement = home_army < jnp.maximum(3, reserve - 6)
        enemy_general = obs.opponent_cells & obs.generals
        wins_now = (
            (base[0] == 0)
            & _neighbors(enemy_general, False)[r, c, direction]
            & (touch | (sent > _neighbors(a, 0)[r, c, direction]))
        )
        # Preserve v2's immediate rescue/third-tile tactics and winning captures.
        emergency = (imminent > home_army) | (touch & (imminent > 0))
        override = active & (would_deplete | need_reinforcement) & ~wins_now & ~emergency
        # Constrain actual v2-scored moves, then select another useful campaign
        # action instead of repeatedly vetoing the first choice and passing.
        retains_reserve = ~general[..., None, None] | (a[..., None, None] - moved >= jnp.maximum(reserve, imminent))
        alternatives = jnp.where(retains_reserve, campaign_scores, -1e9)
        best = jnp.argmax(alternatives)
        best_cell = best // 8
        productive = jnp.array([0, best_cell // w, best_cell % w, (best // 2) % 4, best % 2], jnp.int32)
        productive = jnp.where(jnp.max(alternatives) > 0, productive, jnp.array([1, 0, 0, 0, 0], jnp.int32))
        recall = need_reinforcement & (defensive[0] == 0)
        fallback = jnp.where(recall, defensive, productive)
        action = jnp.where(override, fallback, base)
        half_sortie = (
            override
            & outgoing
            & (action[0] == 0)
            & jnp.all(action[1:4] == base[1:4])
            & (action[4] == 1)
            & (base[4] == 0)
        )
        updated = SentinelMemory(threats, ages, obs.timestep, reserve, until)
        telemetry = dict(telemetry)
        telemetry.update(
            v2_general_reserve=telemetry["general_reserve"],
            general_reserve=reserve,
            defense_active=active,
            defense_override=override,
            defense_recall=override & recall,
            defense_half_sortie=half_sortie,
            defense_productive_alternative=override & ~recall & (action[0] == 0),
            fallback_score=jnp.max(alternatives),
            visible_reserve=visible_reserve,
            remembered_reserve=fog_reserve,
            remembered_cells=jnp.sum((threats > 0) & ~observed),
            maximum_threat_age=jnp.max(ages),
            defense_until=until,
            building=action[0] == 2,
        )
        return action, updated, telemetry

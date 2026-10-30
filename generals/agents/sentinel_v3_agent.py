"""Experimental stateful defense layered over the unchanged Sentinel v2 policy.

Remembered armies describe possible threats, never fabricated observation cells.
The max envelope avoids adding copies of a single uncertain army together.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs

from .sentinel_agent import FrozenSentinelAgent, _distance, _neighbors


# Deliberate frozen-v2 scoring copy: only the extra scores return differs.
# Keep action-parity tests when changing this local extraction; never edit v2.
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
        self.baseline = FrozenSentinelAgent(
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
        base, telemetry, campaign_scores = self.baseline.campaign_decision(obs, key)
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

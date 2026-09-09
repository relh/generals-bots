"""One bounded offensive collection/deployment plan over frozen Sentinel v6.

Only owned transport and currently visible territorial objectives are planned.
Delivered force and home coverage are current-observation estimates, not a
prediction of enemy movement, merging, or unseen armies. One target and one
six-step owned-route dynamic program bound the additional inference work.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance, _neighbors
from .sentinel_v5_agent import _home_growth
from .sentinel_v6_agent import DefenderMemory, SentinelV6Agent, _destination

_INF = 1e6
_COLLECTION_STEPS = 6


class OffensiveMemory(NamedTuple):
    defense: DefenderMemory
    packet: jax.Array
    rally: jax.Array
    objective: jax.Array
    phase: jax.Array
    remaining: jax.Array
    expires: jax.Array
    last_turn: jax.Array
    expected_army: jax.Array


def _collection_route(armies, eligible, target):
    """Largest delivery along a simple shortest owned route, each cell once."""
    distance = _distance(eligible, target)
    initial = jnp.where(target, armies.astype(jnp.float32), -_INF)

    def body(_, carry):
        delivered, largest = carry
        values = _neighbors(delivered, -_INF)
        allowed = _neighbors(distance, _INF) == distance[..., None] - 1
        choice = jnp.argmax(jnp.where(allowed, values, -_INF), axis=-1)
        best = jnp.take_along_axis(values, choice[..., None], axis=-1)[..., 0]
        biggest = jnp.take_along_axis(_neighbors(largest, _INF), choice[..., None], axis=-1)[..., 0]
        reachable = eligible & (best > -_INF / 2)
        return (
            jnp.where(target, armies, jnp.where(reachable, armies - 1 + best, -_INF)),
            jnp.where(target, armies, jnp.where(reachable, jnp.maximum(armies, biggest), _INF)),
        )

    delivered, largest = jax.lax.fori_loop(0, _COLLECTION_STEPS, body, (initial, jnp.where(target, armies, _INF)))
    allowed = _neighbors(distance, _INF) == distance[..., None] - 1
    direction = jnp.argmax(jnp.where(allowed, _neighbors(delivered, -_INF), -_INF), axis=-1)
    return distance, delivered, largest, direction


def _preserves_home(obs, action, terrain, home, home_distance, touch_turn):
    """Do not uncover a covered visible route, or worsen an uncovered route."""
    armies, mine = obs.armies, obs.owned_cells
    width = armies.shape[1]
    source = action[1] * width + action[2]
    dest = _destination(action, width, armies.shape)
    moving = jnp.maximum(armies.reshape(-1)[source] - 1, 0)
    target_army = armies.reshape(-1)[dest]
    target_owned = mine.reshape(-1)[dest]
    after = armies.reshape(-1).at[source].set(1)
    after = (
        after.at[dest]
        .set(jnp.where(target_owned, target_army + moving, jnp.maximum(moving - target_army, 0)))
        .reshape(armies.shape)
    )
    new_owned = mine.reshape(-1).at[dest].set(True).reshape(mine.shape)
    before_cost = _distance(terrain, home, 1 + jnp.where(mine | obs.neutral_cells, armies, 0))
    after_cost = _distance(terrain, home, 1 + jnp.where(new_owned | obs.neutral_cells, after, 0))
    touch = obs.timestep + home_distance - 1 >= touch_turn if touch_turn is not None else jnp.zeros_like(mine)
    adjustment = jnp.where(touch, -jnp.sum(jnp.where(home, armies, 0)), _home_growth(obs.timestep, home_distance))
    before_cost += adjustment
    after_cost += adjustment
    enemy = obs.opponent_cells.reshape(-1).at[dest].set(False).reshape(mine.shape)
    relevant = enemy & (home_distance <= 10)
    adequate = jnp.where(armies <= before_cost, armies <= after_cost, after_cost >= before_cost)
    return jnp.all(~relevant | adequate)


class SentinelV7Agent:
    """Disabled concentration returns exact v6 actions, telemetry and memory."""

    def __init__(
        self, id="Sentinel-v7", *, build_castles=False, deathtouch_turn=None, max_turns=1200, concentrate_armies=True
    ):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.concentrate_armies = concentrate_armies
        self.base = SentinelV6Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        return OffensiveMemory(self.base.initial_memory(shape), *(jnp.int32(x) for x in (-1, -1, -1, 0, 0, -1, -1, 0)))

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        base_action, defense, telemetry = self.base.step(obs, key, memory.defense)
        empty = self.initial_memory(obs.armies.shape)._replace(
            defense=defense, last_turn=obs.timestep.astype(jnp.int32)
        )
        if not self.concentrate_armies:
            return base_action, empty, telemetry

        a, mine = obs.armies, obs.owned_cells
        h, w = a.shape
        cells = jnp.arange(a.size).reshape(a.shape)
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        local_force = jnp.max(_neighbors(jnp.where(obs.opponent_cells, jnp.maximum(a - 1, 0), 0), 0), axis=-1)
        eligible = mine & ~home & (local_force == 0)
        surplus = jnp.where(eligible, jnp.maximum(a - 1, 0), 0)
        local_pool = jax.lax.reduce_window(surplus, 0, jax.lax.add, (7, 7), (1, 1), "SAME")
        active_touch = obs.timestep >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        target_force = jnp.where(obs.generals & active_touch, 0, a)
        target_candidates = obs.opponent_cells & terrain & (local_pool > target_force + 2)
        target_scores = jnp.where(
            target_candidates, a + obs.castles * 20 + obs.generals * 100 - home_distance * 0.1, -_INF
        )
        new_objective = jnp.argmax(target_scores)
        had_plan = memory.phase > 0
        objective = jnp.where(had_plan, memory.objective, new_objective)
        objective = jnp.clip(objective, 0, a.size - 1)
        target = cells == objective
        attack_cells = terrain & (mine | target) & ~home
        attack_distance = _distance(attack_cells, target)
        stages = eligible & (attack_distance >= 2) & (attack_distance <= 3) & (home_distance >= 3)
        rally_score = jnp.where(stages, a + local_pool * 0.15 - attack_distance * 2, -_INF)
        rally = jnp.clip(jnp.where(had_plan, memory.rally, jnp.argmax(rally_score)), 0, a.size - 1)
        rally_mask = cells == rally
        distance, delivered, largest, direction = _collection_route(a, eligible, rally_mask)
        attack_eta = attack_distance.reshape(-1)[rally]
        counter_force = local_force.reshape(-1)[objective]
        target_is_general = obs.generals.reshape(-1)[objective]
        target_army = a.reshape(-1)[objective]
        projected_touch = (
            obs.timestep + distance + attack_eta - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(mine)
        )
        # Owned path garrisons can only improve this conservative delivery bound.
        required = jnp.where(
            target_is_general & projected_touch, attack_eta + 1, target_army + counter_force + attack_eta + 1
        )
        source_ok = (
            eligible
            & (distance >= 1)
            & (distance <= _COLLECTION_STEPS)
            & (delivered >= largest * 1.5)
            & (delivered >= required)
        )
        source_score = jnp.where(source_ok, delivered - distance - attack_eta, -_INF)
        packet = jnp.clip(jnp.where(had_plan, memory.packet, jnp.argmax(source_score)), 0, a.size - 1)
        collecting = ~had_plan | (memory.phase == 1)
        remaining = jnp.where(collecting, distance.reshape(-1)[packet], attack_distance.reshape(-1)[packet])
        deploy_direction = jnp.argmin(_neighbors(attack_distance, _INF), axis=-1).reshape(-1)[packet]
        move_direction = jnp.where(collecting, direction.reshape(-1)[packet], deploy_direction)
        planned = jnp.array([0, packet // w, packet % w, move_direction, 0], jnp.int32)
        destination = _destination(planned, w, a.shape)
        destination_mine = mine.reshape(-1)[destination]
        packet_army = a.reshape(-1)[packet]
        force_eta = jnp.where(collecting, remaining + attack_eta, remaining)
        touch_on_attack = (
            obs.timestep + force_eta - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.array(False)
        )
        need = jnp.where(
            target_is_general & touch_on_attack,
            jnp.where(collecting, attack_eta, remaining) + 1,
            target_army + counter_force + jnp.where(collecting, attack_eta, remaining) + 1,
        )
        enough = jnp.where(collecting, delivered.reshape(-1)[packet] >= need, packet_army >= need)
        consistent = (
            mine.reshape(-1)[packet]
            & ~home.reshape(-1)[packet]
            & (packet_army > 1)
            & (packet_army >= memory.expected_army)
        )
        continuous = (memory.last_turn + 1 == obs.timestep) & (obs.timestep <= memory.expires)
        objective_valid = obs.opponent_cells.reshape(-1)[objective]
        legal_step = (remaining >= 1) & (remaining < _INF / 2) & (destination_mine | (destination == objective))
        capture = (packet_army - 1 > a.reshape(-1)[destination]) | (
            target_is_general & active_touch & (destination == objective)
        )
        legal_step &= destination_mine | capture
        priority = (
            (memory.defense.defender >= 0)
            | (defense.defender >= 0)
            | telemetry["intercept_guard"]
            | (telemetry["adjacent_threat"] > 0)
        )
        priority |= telemetry["general_reserve"] > telemetry["general_army"]
        base_dest = _destination(base_action, w, a.shape)
        base_source = base_action[1] * w + base_action[2]
        base_moving = jnp.where(base_action[4] == 1, a.reshape(-1)[base_source] // 2, a.reshape(-1)[base_source] - 1)
        winning = (
            (base_action[0] == 0)
            & (obs.opponent_cells & obs.generals).reshape(-1)[base_dest]
            & ((base_moving > a.reshape(-1)[base_dest]) | active_touch)
        )
        priority |= winning
        preserves = _preserves_home(obs, planned, terrain, home, home_distance, self.deathtouch_turn)
        start = ~had_plan & jnp.any(target_candidates) & jnp.any(stages) & jnp.any(source_ok) & (base_action[0] != 2)
        continue_plan = had_plan & continuous & consistent & (remaining <= memory.remaining)
        continue_plan &= jnp.where(collecting, eligible.reshape(-1)[packet] & eligible.reshape(-1)[rally], True)
        use = (start | continue_plan) & objective_valid & legal_step & enough & preserves & ~priority
        action = jnp.where(use, planned, base_action)
        attacking = use & (destination == objective)
        next_phase = jnp.where(collecting & (destination != rally), 1, 2)
        next_remaining = jnp.where(collecting & (destination == rally), attack_eta, remaining - 1)
        next_memory = OffensiveMemory(
            defense,
            destination.astype(jnp.int32),
            rally.astype(jnp.int32),
            objective.astype(jnp.int32),
            next_phase.astype(jnp.int32),
            next_remaining.astype(jnp.int32),
            jnp.where(had_plan, memory.expires, obs.timestep + remaining + attack_eta + 1).astype(jnp.int32),
            obs.timestep.astype(jnp.int32),
            (packet_army - 1 + a.reshape(-1)[destination]).astype(jnp.int32),
        )
        memory = jax.tree.map(lambda planned, blank: jnp.where(use & ~attacking, planned, blank), next_memory, empty)
        telemetry = telemetry | dict(
            offense_started=use & ~had_plan,
            offense_continued=use & had_plan,
            offense_collecting=use & collecting,
            offense_deploying=use & ~collecting,
            offense_attack_issued=attacking,
            offense_override=jnp.any(action != base_action),
            offense_released=had_plan & ~use,
            offense_defense_priority=priority,
            offense_home_coverage_safe=preserves,
            offense_feasible=start | continue_plan,
            offense_objective=jnp.where(use, objective, -1),
            offense_rally=jnp.where(use, rally, -1),
            offense_delivered=jnp.where(use & collecting, delivered.reshape(-1)[packet], 0),
            offense_required=jnp.where(use, need, 0),
            offense_remaining=jnp.where(use, remaining, 0),
        )
        telemetry["building"] = action[0] == 2
        return action, memory, telemetry

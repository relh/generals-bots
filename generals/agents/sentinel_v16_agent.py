"""Experimental same-objective feeder routes inside the frozen V10 stack.

The inner campaign retains V8 selection except when a new collection displaces
an actual V6 full owned transfer to the same visible objective. A feeder must
provide more post-capture force AND more force per action. This is a declared
value hypothesis, not dominance over an earlier capture or a safety guarantee.

Nine-step shortest-owned-route DP charges each departure once and follows its
maximizing suffix. The collector comparison includes all best deployment
contributions; overlap with collection may overcredit the collector, making
admission conservative against the feeder. Neither estimate predicts growth,
opponent responses or hidden armies. Current first-step home coverage retains
V8's approximate visible-threat check; future donations are revalidated.

Phase3 reuses OffensiveMemory. V9/V10 remain outer layers, so pursuit or defense
can clear unissued transport. Feeder events describe the inner proposal: mask
with actual_v8_action_issued and mobilization_parent_action_issued for final
execution. Other offense events describe the actual inner campaign action.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance, _neighbors
from .sentinel_v6_agent import _destination
from .sentinel_v7_agent import OffensiveMemory, _collection_route, _preserves_home
from .sentinel_v8_agent import SentinelV8Agent, _select_collection
from .sentinel_v10_agent import SentinelV10Agent

_INF = 1e6
_COLLECTION_STEPS = 6
_DIRECT_STEPS = 9


def _route_residual(armies, owned, home, target, distance):
    """Best post-capture force along a strictly decreasing owned route.

    A zero-army intermediate is excluded: reaching it with two armies could
    stall before the later donors whose force would certify the route.
    """
    endpoint = -armies.astype(jnp.float32)
    initial = jnp.where(target, endpoint, -_INF)
    transit = owned & ~home & (armies >= 1)

    def body(_, residual):
        neighbors = _neighbors(residual, -_INF)
        progress = _neighbors(distance, _INF) == distance[..., None] - 1
        best = jnp.max(jnp.where(progress, neighbors, -_INF), axis=-1)
        value = armies - 1 + best
        return jnp.where(target, endpoint, jnp.where(transit & (best > -_INF / 2), value, -_INF))

    residual = jax.lax.fori_loop(0, _DIRECT_STEPS, body, initial)
    progress = _neighbors(distance, _INF) == distance[..., None] - 1
    direction = jnp.argmax(jnp.where(progress, _neighbors(residual, -_INF), -_INF), axis=-1)
    return residual, direction


class _FeederCampaign(SentinelV8Agent):
    """Frozen V8 step with one additional route candidate and phase3 transport."""

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        if not self.concentrate_armies or not (self.cheapest_collection or self.direct_deployment):
            return self.reference.step(obs, key, memory)

        incoming = memory
        base_action, defense, telemetry = self.base.step(obs, key, memory.defense)
        empty = self.initial_memory(obs.armies.shape)._replace(
            defense=defense, last_turn=obs.timestep.astype(jnp.int32)
        )

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
        feeder_active = memory.phase == 3
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
        collection_packet = _select_collection(source_ok, distance, delivered, attack_eta, self.cheapest_collection)
        collection_available = jnp.any(source_ok) & jnp.any(stages)
        collection_eta = jnp.where(collection_available, distance.reshape(-1)[collection_packet] + attack_eta, _INF)
        # The ready packet pays each route departure; future friendly garrisons
        # are not credited before their transfers have actually been observed.
        direct_touch = (
            obs.timestep + attack_distance - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(mine)
        )
        direct_required = jnp.where(
            target_is_general & direct_touch,
            attack_distance + 1,
            target_army + counter_force + attack_distance + 1,
        )
        direct_ok = (
            mine
            & ~home
            & (a > 1)
            & (attack_distance >= 1)
            & (attack_distance <= _DIRECT_STEPS)
            & (a >= direct_required)
        )
        direct_eta = jnp.min(jnp.where(direct_ok, attack_distance, _INF))
        direct_packet = jnp.argmax(jnp.where(direct_ok & (attack_distance == direct_eta), a, -_INF))
        direct_available = self.direct_deployment & jnp.any(direct_ok) & jnp.any(target_candidates)
        choose_direct = ~had_plan & direct_available & (direct_eta < collection_eta)
        new_packet = jnp.where(choose_direct, direct_packet, collection_packet)
        packet = jnp.clip(jnp.where(had_plan, memory.packet, new_packet), 0, a.size - 1)
        # Force the reference's ACTUAL first branch, then maximize its suffix.
        residual, route_direction = _route_residual(a, mine, home, target, attack_distance)
        reference_source = jnp.clip(base_action[1] * w + base_action[2], 0, a.size - 1)
        reference_destination = _destination(base_action, w, a.shape)
        reference_eta = attack_distance.reshape(-1)[reference_source]
        reference_residual = (
            a.reshape(-1)[reference_source] - 1 + residual.reshape(-1)[reference_destination]
        )
        collector_residual = (
            delivered.reshape(-1)[collection_packet]
            + residual.reshape(-1)[rally] - a.reshape(-1)[rally]
        )
        collector_move = jnp.array([
            0, collection_packet // w, collection_packet % w,
            direction.reshape(-1)[collection_packet], 0,
        ], jnp.int32)
        # Admission replaces an actually admissible collection, not a proposal
        # that the original V8 home check would already reject.
        collector_safe = _preserves_home(
            obs, collector_move, terrain, home, home_distance, self.deathtouch_turn,
        )
        reference_safe = _preserves_home(
            obs, base_action, terrain, home, home_distance, self.deathtouch_turn,
        )
        reference_owned = (
            (base_action[0] == 0) & (base_action[4] == 0)
            & mine.reshape(-1)[reference_source] & ~home.reshape(-1)[reference_source]
            & (a.reshape(-1)[reference_source] > 1)
            & mine.reshape(-1)[reference_destination]
            & ~home.reshape(-1)[reference_destination]
            & (attack_distance.reshape(-1)[reference_destination] == reference_eta - 1)
            & (reference_eta >= 2) & (reference_eta <= _DIRECT_STEPS)
            & (residual.reshape(-1)[reference_destination] > -_INF / 2)
        )
        feeder_available = (
            ~had_plan & ~choose_direct & collection_available & reference_owned
            & jnp.any(target_candidates) & ~target_is_general
            & ~jnp.any(obs.opponent_cells & obs.generals)
            & collector_safe & reference_safe
            & (collector_residual > 0) & (reference_residual > counter_force)
        )
        choose_feeder = (
            feeder_available & (reference_residual > collector_residual)
            & (reference_residual * collection_eta > collector_residual * reference_eta)
        )
        feeder = feeder_active | choose_feeder
        packet = jnp.where(choose_feeder, reference_source, packet)
        # Keep the exact memory shape; deployment does not consume a rally.
        rally = jnp.where(choose_direct | choose_feeder, packet, rally)
        collecting = jnp.where(had_plan, memory.phase == 1, ~(choose_direct | choose_feeder))
        remaining = jnp.where(collecting, distance.reshape(-1)[packet], attack_distance.reshape(-1)[packet])
        deploy_direction = jnp.argmin(_neighbors(attack_distance, _INF), axis=-1).reshape(-1)[packet]
        move_direction = jnp.where(collecting, direction.reshape(-1)[packet], deploy_direction)
        move_direction = jnp.where(feeder_active, route_direction.reshape(-1)[packet], move_direction)
        move_direction = jnp.where(choose_feeder, base_action[3], move_direction)
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
        feeder_residual = jnp.where(choose_feeder, reference_residual, residual.reshape(-1)[packet])
        enough = jnp.where(feeder, feeder_residual > counter_force, enough)
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
        # Builds and newly visible generals take priority even mid-feeder route.
        priority |= feeder & ((base_action[0] == 2) | jnp.any(obs.opponent_cells & obs.generals))
        preserves = _preserves_home(obs, planned, terrain, home, home_distance, self.deathtouch_turn)
        start = ~had_plan & jnp.any(target_candidates) & (collection_available | choose_direct) & (base_action[0] != 2)
        continue_plan = had_plan & continuous & consistent & (remaining <= memory.remaining)
        continue_plan &= jnp.where(collecting, eligible.reshape(-1)[packet] & eligible.reshape(-1)[rally], True)
        continue_plan &= ~feeder_active | (remaining == memory.remaining)
        use = (start | continue_plan) & objective_valid & legal_step & enough & preserves & ~priority
        action = jnp.where(use, planned, base_action)
        attacking = use & (destination == objective)
        next_phase = jnp.where(feeder, 3, jnp.where(collecting & (destination != rally), 1, 2))
        next_remaining = jnp.where(collecting & (destination == rally), attack_eta, remaining - 1)
        next_memory = OffensiveMemory(
            defense,
            destination.astype(jnp.int32),
            rally.astype(jnp.int32),
            objective.astype(jnp.int32),
            next_phase.astype(jnp.int32),
            next_remaining.astype(jnp.int32),
            jnp.where(had_plan, memory.expires, obs.timestep + force_eta + 1).astype(jnp.int32),
            obs.timestep.astype(jnp.int32),
            (packet_army - 1 + a.reshape(-1)[destination]).astype(jnp.int32),
        )
        memory = jax.tree.map(lambda planned, blank: jnp.where(use & ~attacking, planned, blank), next_memory, empty)
        telemetry = telemetry | dict(
            feeder_available=feeder_available,
            feeder_chosen=choose_feeder | feeder_active,
            feeder_issued=use & feeder,
            feeder_differs_from_reference=use & feeder & jnp.any(action != base_action),
            feeder_started=use & choose_feeder,
            feeder_continued=use & feeder_active,
            feeder_released=feeder_active & ~use,
            feeder_eta=jnp.where(feeder, force_eta, reference_eta),
            feeder_residual=jnp.where(feeder, feeder_residual, reference_residual),
            feeder_collector_eta=collection_eta,
            feeder_collector_residual_upper=collector_residual,
            feeder_home_coverage_safe=preserves,
            feeder_reference_home_safe=reference_safe,
            feeder_collector_home_safe=collector_safe,
            feeder_reference_defender_active=defense.defender >= 0,
            feeder_objective=objective,
            feeder_packet=packet,
            feeder_abort_gap=feeder_active & (incoming.last_turn + 1 != obs.timestep),
            feeder_abort_expired=feeder_active & (obs.timestep > incoming.expires),
            feeder_abort_inconsistent=feeder_active & ~consistent,
            feeder_abort_objective=feeder_active & ~objective_valid,
            feeder_abort_route=feeder_active & (~legal_step | (remaining != incoming.remaining)),
            feeder_abort_force=feeder_active & ~enough,
            feeder_abort_home=feeder_active & ~preserves,
            feeder_abort_priority=feeder_active & priority,
        )
        for label, candidate in (("collector", collector_move), ("reference", base_action), ("proposal", planned)):
            for index, field in enumerate(("kind", "row", "column", "direction", "split")):
                telemetry[f"feeder_{label}_{field}"] = candidate[index]
        telemetry = telemetry | dict(
            offense_direct_started=use & choose_direct,
            offense_direct_chosen=choose_direct,
            offense_direct_available=direct_available & ~had_plan,
            offense_direct_eta=jnp.where(direct_available & ~had_plan, direct_eta, -1),
            offense_collection_eta=jnp.where(collection_available & ~had_plan, collection_eta, -1),
            offense_cheapest_collection=jnp.array(self.cheapest_collection),
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


class SentinelV16Agent:
    """Same-front feeder candidate; disabled is exact frozen V10/native19."""

    def __init__(
        self, id="Sentinel-v16", *, build_castles=False, deathtouch_turn=None,
        max_turns=1200, feed_front=True,
    ):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.feed_front = feed_front
        self.base = SentinelV10Agent(
            build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns,
        )
        if feed_front:
            self.base.base.base = _FeederCampaign(
                build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns,
            )

    def initial_memory(self, shape):
        return self.base.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        return self.base.step(obs, key, memory)

"""Experimental visible-path interception over the frozen v2 campaign.

Plans use at most twelve interception targets and ten owned-route steps. They
are estimates for a currently visible stack, not guarantees against detours,
enemy merges/growth, multiple simultaneous invaders, or future fog movement.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .agent import Agent
from .sentinel_agent import SentinelAgent, _build_prices, _distance, _neighbors
from .sentinel_v3_agent import _campaign_decision

_HORIZON = 10
_TARGETS_PER_END = 6
_INF = 1e6


def _home_growth(timestep, arrival):
    """Guaranteed structure/land production before the future attacking move."""
    before_attack = jnp.maximum(timestep, timestep + arrival - 1)
    return before_attack // 2 - timestep // 2 + before_attack // 50 - timestep // 50


def _after_own_action(obs, action):
    """Observable garrisons after our proposed action, before an enemy reply."""
    kind, row, col, direction, split = action
    dr = jnp.array([-1, 1, 0, 0])[direction]
    dc = jnp.array([0, 0, -1, 1])[direction]
    rr, cc = jnp.clip(row + dr, 0, obs.armies.shape[0] - 1), jnp.clip(col + dc, 0, obs.armies.shape[1] - 1)
    moved = jnp.where(split == 1, obs.armies[row, col] // 2, obs.armies[row, col] - 1)
    moved = jnp.where(kind == 0, jnp.maximum(moved, 0), 0)
    price = _build_prices(obs.owned_cells & (obs.generals | obs.castles))
    spent = jnp.where(kind == 2, price[row, col], 0)
    armies = obs.armies.at[row, col].add(-moved - spent)
    armies = armies.at[rr, cc].add(jnp.where(obs.owned_cells[rr, cc], moved, 0))
    captured = (kind == 0) & ~obs.owned_cells[rr, cc] & (moved > obs.armies[rr, cc])
    armies = armies.at[rr, cc].set(jnp.where(captured, moved - obs.armies[rr, cc], armies[rr, cc]))
    owned = obs.owned_cells.at[rr, cc].set(obs.owned_cells[rr, cc] | captured)
    removed = captured & obs.opponent_cells[rr, cc]
    return armies, owned, removed, rr, cc, moved


def _owned_route_plan(obs, terrain, home, enemy_distance, corridor, target):
    """Best simple shortest owned route per source, without double-counted donors.

    The target keeps its existing garrison. Other garrisons lying in the enemy's
    homeward corridor already contribute stationary attrition, so their transfer
    is not counted as new defense. Strictly decreasing target distance prevents
    revisiting a cell and repeatedly crediting its army.
    """
    route_cells = obs.owned_cells & (~home | target)
    distance = _distance(terrain & route_cells, target)
    donation = jnp.where(corridor, 0, jnp.maximum(obs.armies - 1, 0))
    initial = jnp.where(target, 0.0, -_INF)
    initial_army = jnp.where(target, obs.armies.astype(jnp.float32), -_INF)
    initial_deadline = jnp.where(target, enemy_distance, -_INF)

    def body(_, carry):
        extra, army, deadline = carry
        next_extra = _neighbors(extra, -_INF)
        next_army = _neighbors(army, -_INF)
        next_deadline = _neighbors(deadline, -_INF)
        allowed = (_neighbors(distance, _INF) == distance[..., None] - 1) & (next_deadline >= distance[..., None])
        choices = jnp.where(allowed, next_extra, -_INF)
        direction = jnp.argmax(choices, axis=-1)

        def take(values):
            return jnp.take_along_axis(values, direction[..., None], axis=-1)[..., 0]

        available = route_cells & (jnp.max(choices, axis=-1) > -_INF / 2)
        return (
            jnp.where(target, 0, jnp.where(available, donation + take(next_extra), -_INF)),
            jnp.where(target, obs.armies, jnp.where(available, obs.armies - 1 + take(next_army), -_INF)),
            jnp.where(
                target,
                enemy_distance,
                jnp.where(available, jnp.minimum(enemy_distance + distance, take(next_deadline)), -_INF),
            ),
        )

    extra, army, deadline = jax.lax.fori_loop(0, _HORIZON, body, (initial, initial_army, initial_deadline))
    next_extra = _neighbors(extra, -_INF)
    allowed = (_neighbors(distance, _INF) == distance[..., None] - 1) & (
        _neighbors(deadline, -_INF) >= distance[..., None]
    )
    direction = jnp.argmax(jnp.where(allowed, next_extra, -_INF), axis=-1)
    return distance, extra, army, deadline, direction


class SentinelV5Agent(Agent):
    """Stateless candidate with an exact-v2 disabled ablation."""

    def __init__(
        self, id="Sentinel-v5", *, build_castles=False, deathtouch_turn=None, max_turns=1200, intercept_threats=True
    ):
        super().__init__(id)
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.intercept_threats = intercept_threats
        self.campaign = SentinelAgent(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    @partial(jax.jit, static_argnums=0)
    def act(self, obs, key):
        return self.decision(obs, key)[0]

    @partial(jax.jit, static_argnums=0)
    def decision(self, obs, key):
        base, telemetry, scores = _campaign_decision(self.campaign, obs, key)
        defense = dict(
            intercept_override=jnp.array(False),
            intercept_guard=jnp.array(False),
            intercept_feasible=jnp.array(False),
            intercept_threat_index=jnp.int32(-1),
            intercept_target_index=jnp.int32(-1),
            intercept_enemy_eta=jnp.float32(0),
            intercept_friendly_eta=jnp.float32(0),
            intercept_new_defense=jnp.float32(0),
            intercept_required_defense=jnp.float32(0),
            intercept_arrival_army=jnp.float32(0),
            intercept_home_deficit=jnp.float32(0),
            intercept_projected_deathtouch=jnp.array(False),
        )
        if not self.intercept_threats:
            return base, telemetry | defense

        mine, a = obs.owned_cells, obs.armies
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        post, post_owned, removed, rr, cc, moved = _after_own_action(obs, base)
        projected_touch = (
            obs.timestep + home_distance - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(mine)
        )
        defenders = mine | obs.neutral_cells
        post_home_cost = _distance(terrain, home, 1 + jnp.where(post_owned | obs.neutral_cells, post, 0))
        post_home_cost -= projected_touch * jnp.sum(jnp.where(home, post, 0))
        home_growth = jnp.where(projected_touch, 0, _home_growth(obs.timestep, home_distance))
        post_home_cost += home_growth
        live_enemy = obs.opponent_cells.at[rr, cc].set(obs.opponent_cells[rr, cc] & ~removed)
        credible = live_enemy & (home_distance <= _HORIZON) & (a > post_home_cost)
        earliest = jnp.min(jnp.where(credible, home_distance, _INF))
        threat_index = jnp.argmax(jnp.where(credible & (home_distance == earliest), a, -1))
        winning = (
            (base[0] == 0)
            & obs.opponent_cells[rr, cc]
            & obs.generals[rr, cc]
            & (telemetry["deathtouch_active"] | (moved > a[rr, cc]))
        )

        def plan(_):
            h, w = a.shape
            index_grid = jnp.arange(a.size).reshape(a.shape)
            enemy = index_grid == threat_index
            enemy_army = a.reshape(-1)[threat_index]
            eta_home = home_distance.reshape(-1)[threat_index]
            touch = projected_touch.reshape(-1)[threat_index]
            enemy_distance = _distance(terrain, enemy)
            costs = 1 + jnp.where(defenders, a, 0)
            current_home_cost = _distance(terrain, home, costs) - touch * jnp.sum(jnp.where(home, a, 0))
            current_home_cost += home_growth.reshape(-1)[threat_index]
            # Reverse node-entry costs: distance(i -> enemy) + cost(i) - cost(enemy).
            enemy_cost = _distance(terrain, enemy, costs) + costs - costs.reshape(-1)[threat_index]
            current_cost = current_home_cost.reshape(-1)[threat_index]
            corridor = (enemy_cost + current_home_cost <= current_cost) & terrain
            target_mask = (
                mine
                & corridor
                & (enemy_distance + home_distance <= eta_home)
                & (enemy_distance > 0)
                & ~(home & touch)
                & (enemy_cost + current_home_cost < enemy_army)
            )
            # Bounded front and home candidate sets; duplicates are harmless.
            front = jnp.argsort(jnp.where(target_mask, enemy_distance, _INF).reshape(-1))[:_TARGETS_PER_END]
            rear = jnp.argsort(jnp.where(target_mask, home_distance, _INF).reshape(-1))[:_TARGETS_PER_END]
            indices = jnp.concatenate((front, rear))
            targets = index_grid[None] == indices[:, None, None]
            distance, extra, arrival_army, deadline, direction = jax.vmap(
                lambda target: _owned_route_plan(obs, terrain, home, enemy_distance, corridor, target)
            )(targets)
            enemy_eta = enemy_distance.reshape(-1)[indices]
            required = jnp.maximum(
                0, enemy_army - enemy_cost.reshape(-1)[indices] - current_home_cost.reshape(-1)[indices]
            )
            eligible = (
                mine[None]
                & ~home[None]
                & (a[None] > 1)
                # An adjacent enemy could chase and capture the source before
                # its reinforcing move executes. Leave that tactic to v2.
                & (enemy_distance[None] > 1)
                & (distance >= 1)
                & (distance <= _HORIZON)
                & (distance <= enemy_eta[:, None, None])
                & (distance <= deadline)
                & (extra >= required[:, None, None])
                & (required[:, None, None] > 0)
                & (extra > 0)
                & target_mask.reshape(-1)[indices, None, None]
            )
            # Prefer a cheap, timely intervention over recalling a remote army.
            rank = (
                distance * 100 + enemy_eta[:, None, None] * 10 + jnp.maximum(extra - required[:, None, None], 0) * 0.01
            )
            choice = jnp.argmin(jnp.where(eligible, rank, _INF).reshape(-1))
            target_slot, source = choice // a.size, choice % a.size
            planned = jnp.array([0, source // w, source % w, direction.reshape(-1)[choice], 0], jnp.int32)
            feasible = jnp.any(eligible)

            # A currently adequate screen can be endangered by the base move.
            # Protect its donated troops while selecting another useful v2 move.
            moved_all = jnp.stack((a - 1, a // 2), axis=-1)[..., None, :]
            clearance = enemy_cost + current_home_cost - enemy_army
            safe_scores = jnp.where(moved_all <= clearance[..., None, None], scores, -1e9)
            best = jnp.argmax(safe_scores)
            cell = best // 8
            guard_action = jnp.array(
                [jnp.max(safe_scores) <= 0, cell // w, cell % w, (best // 2) % 4, best % 2], jnp.int32
            )
            guard = current_cost >= enemy_army
            action = jnp.where(guard, guard_action, jnp.where(feasible, planned, base))
            override = jnp.any(action != base)
            return action, defense | dict(
                intercept_override=override,
                intercept_guard=guard & override,
                intercept_feasible=feasible,
                intercept_threat_index=threat_index.astype(jnp.int32),
                intercept_target_index=jnp.where(feasible, indices[target_slot], -1).astype(jnp.int32),
                intercept_enemy_eta=jnp.where(feasible, enemy_eta[target_slot], 0),
                intercept_friendly_eta=jnp.where(feasible, distance.reshape(-1)[choice], 0),
                intercept_new_defense=jnp.where(feasible, extra.reshape(-1)[choice], 0),
                intercept_required_defense=jnp.where(feasible, required[target_slot], 0),
                intercept_arrival_army=jnp.where(feasible, arrival_army.reshape(-1)[choice], 0),
                intercept_home_deficit=jnp.maximum(0, enemy_army - current_cost),
                intercept_projected_deathtouch=touch,
            )

        action, defense = jax.lax.cond(jnp.any(credible) & ~winning, plan, lambda _: (base, defense), operand=None)
        telemetry = telemetry | defense
        telemetry["building"] = action[0] == 2
        return action, telemetry

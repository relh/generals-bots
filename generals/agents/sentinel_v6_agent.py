"""Experimental commitment to our defender, without remembered enemy armies.

A short owned-route plan progresses toward a fixed target. Arrived defenders
are not repeatedly recruited as fresh donations while the current visible
threat still justifies their position. Coverage remains an approximate single
visible-stack estimate, with the same detour/merge limitations as v5.
After a held move is excluded, campaign alternatives are not a proof that all
other defensive screens remain adequate. Observed arrival checks are conservative
consistency checks, not evidence that an issued action necessarily executed.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance
from .sentinel_v3_agent import _campaign_decision
from .sentinel_v5_agent import _HORIZON, _INF, SentinelV5Agent, _home_growth, _owned_route_plan


class DefenderMemory(NamedTuple):
    defender: jax.Array
    target: jax.Array
    previous: jax.Array
    distance: jax.Array
    expires: jax.Array
    last_turn: jax.Array
    expected_army: jax.Array


def _destination(action, width, shape):
    row, col, direction = action[1:4]
    dr, dc = jnp.array([-1, 1, 0, 0])[direction], jnp.array([0, 0, -1, 1])[direction]
    return jnp.clip(row + dr, 0, shape[0] - 1) * width + jnp.clip(col + dc, 0, shape[1] - 1)


class SentinelV6Agent:
    """Stateful wrapper; disabled commitment reproduces frozen v5 decisions."""

    def __init__(
        self, id="Sentinel-v6", *, build_castles=False, deathtouch_turn=None, max_turns=1200, commit_defense=True
    ):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.commit_defense = commit_defense
        self.base = SentinelV5Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        del shape
        return DefenderMemory(*(jnp.int32(value) for value in (-1, -1, -1, 0, -1, -1, 0)))

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, telemetry = self.base.decision(obs, key)
        empty = self.initial_memory(obs.armies.shape)._replace(last_turn=obs.timestep.astype(jnp.int32))
        if not self.commit_defense:
            return action, empty, telemetry

        a, mine = obs.armies, obs.owned_cells
        height, width = a.shape
        index_grid = jnp.arange(a.size).reshape(a.shape)
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        defender = jnp.clip(memory.defender, 0, a.size - 1)
        target_index = jnp.clip(memory.target, 0, a.size - 1)
        target = index_grid == target_index
        prior_active = memory.defender >= 0
        army_at_defender = a.reshape(-1)[defender]
        expired = obs.timestep > memory.expires
        observation_gap = memory.last_turn + 1 != obs.timestep
        continuous = ~observation_gap & ~expired
        arrival_consistent = (
            mine.reshape(-1)[defender]
            & mine.reshape(-1)[target_index]
            & ~home.reshape(-1)[defender]
            & (army_at_defender > 1)
            & (army_at_defender >= memory.expected_army)
        )
        valid = prior_active & (memory.target >= 0) & continuous & arrival_consistent
        # Assess current visible need without treating the committed army as a
        # new donor. Enemy locations/forces are never carried across observations.
        without = a.reshape(-1).at[defender].set(1).reshape(a.shape)
        costs_without = 1 + jnp.where(mine | obs.neutral_cells, without, 0)
        without_cost = _distance(terrain, home, costs_without)
        touch = (
            obs.timestep + home_distance - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(mine)
        )
        without_cost += jnp.where(touch, -jnp.sum(jnp.where(home, a, 0)), _home_growth(obs.timestep, home_distance))
        visible_need = obs.opponent_cells & (home_distance <= _HORIZON) & (a > without_cost)
        earliest = jnp.min(jnp.where(visible_need, home_distance, _INF))
        enemy_index = jnp.argmax(jnp.where(visible_need & (home_distance == earliest), a, -1))
        enemy = index_grid == enemy_index
        enemy_distance = _distance(terrain, enemy)
        eta_home = home_distance.reshape(-1)[enemy_index]
        target_eta = enemy_distance.reshape(-1)[target_index]
        target_on_route = target_eta + home_distance.reshape(-1)[target_index] <= eta_home
        current_costs = 1 + jnp.where(mine | obs.neutral_cells, a, 0)
        current_home = _distance(terrain, home, current_costs)
        projected_touch = touch.reshape(-1)[enemy_index]
        current_home += jnp.where(
            projected_touch, -jnp.sum(jnp.where(home, a, 0)), _home_growth(obs.timestep, eta_home)
        )
        from_enemy = _distance(terrain, enemy, current_costs) + current_costs - current_costs.reshape(-1)[enemy_index]
        corridor = from_enemy + current_home <= current_home.reshape(-1)[enemy_index]
        distance, extra, arrival, deadline, direction = _owned_route_plan(
            obs, terrain, home, enemy_distance, corridor, target
        )
        remaining = distance.reshape(-1)[defender]
        required = jnp.maximum(
            0,
            a.reshape(-1)[enemy_index] - from_enemy.reshape(-1)[target_index] - current_home.reshape(-1)[target_index],
        )
        target_allowed = target_on_route & ~(home.reshape(-1)[target_index] & projected_touch)
        arrived = defender == target_index
        progressing = (
            (remaining > 0)
            & (remaining <= memory.distance)
            & (remaining <= target_eta)
            & (remaining <= deadline.reshape(-1)[defender])
            & (enemy_distance.reshape(-1)[defender] > 1)
            & (extra.reshape(-1)[defender] >= required)
        )
        keep = valid & jnp.any(visible_need) & target_allowed & ((arrived & (required <= 0)) | progressing)
        follow = jnp.array([0, defender // width, defender % width, direction.reshape(-1)[defender], 0], jnp.int32)

        campaign_action, campaign_telemetry, scores = _campaign_decision(self.base.campaign, obs, key)
        dest = _destination(campaign_action, width, a.shape)
        source = campaign_action[1] * width + campaign_action[2]
        moved = jnp.where(campaign_action[4] == 1, a.reshape(-1)[source] // 2, a.reshape(-1)[source] - 1)
        winning = (
            (campaign_action[0] == 0)
            & (obs.opponent_cells & obs.generals).reshape(-1)[dest]
            & ((moved > a.reshape(-1)[dest]) | campaign_telemetry["deathtouch_active"])
        )
        immediate = telemetry["adjacent_threat"] > 0
        priority = winning | immediate | telemetry["intercept_guard"]
        keep &= ~priority

        # If the original route is obsolete, do not immediately count the just
        # relocated defender as a fresh donation back to its previous square.
        action_source = action[1] * width + action[2]
        action_dest = _destination(action, width, a.shape)
        planned_defense = telemetry["intercept_feasible"] & (telemetry["intercept_home_deficit"] > 0)
        reverse = (
            valid
            & planned_defense
            & ~telemetry["intercept_guard"]
            & (action[0] == 0)
            & (action_source == defender)
            & (action_dest == memory.previous)
            & ~priority
        )
        hold = keep & arrived
        block_defender = hold | reverse
        other_scores = jnp.where((index_grid == defender)[..., None, None], -1e9, scores)
        best = jnp.argmax(other_scores)
        cell = best // 8
        alternative = jnp.array(
            [jnp.max(other_scores) <= 0, cell // width, cell % width, (best // 2) % 4, best % 2], jnp.int32
        )
        # Keep a profitable build elsewhere and any already-selected v5 move
        # that does not consume the held defender. Guards retain their priority.
        base_uses_defender = (action[0] != 1) & (action_source == defender)
        replacement = jnp.where(base_uses_defender, alternative, action)
        elsewhere_build = (campaign_action[0] == 2) & (source != defender)
        replacement = jnp.where(elsewhere_build & base_uses_defender, campaign_action, replacement)
        chosen = jnp.where(keep & ~arrived, follow, jnp.where(block_defender, replacement, action))
        chosen = jnp.where(winning, campaign_action, chosen)

        dest_chosen = _destination(chosen, width, a.shape)
        chosen_source = chosen[1] * width + chosen[2]
        start = (
            ~keep
            & ~reverse
            & ~priority
            & planned_defense
            & ~telemetry["intercept_guard"]
            & (chosen[0] == 0)
            & mine.reshape(-1)[dest_chosen]
            & (telemetry["intercept_target_index"] >= 0)
        )
        continuing = DefenderMemory(
            jnp.where(arrived, defender, dest_chosen).astype(jnp.int32),
            memory.target,
            jnp.where(arrived, memory.previous, defender).astype(jnp.int32),
            jnp.where(arrived, 0, remaining - 1).astype(jnp.int32),
            memory.expires,
            obs.timestep.astype(jnp.int32),
            jnp.where(arrived, army_at_defender, army_at_defender - 1 + a.reshape(-1)[dest_chosen]).astype(jnp.int32),
        )
        started = DefenderMemory(
            dest_chosen.astype(jnp.int32),
            telemetry["intercept_target_index"].astype(jnp.int32),
            chosen_source.astype(jnp.int32),
            (telemetry["intercept_friendly_eta"] - 1).astype(jnp.int32),
            (obs.timestep + _HORIZON).astype(jnp.int32),
            obs.timestep.astype(jnp.int32),
            (a.reshape(-1)[chosen_source] - 1 + a.reshape(-1)[dest_chosen]).astype(jnp.int32),
        )
        memory = jax.tree.map(
            lambda old, new, blank: jnp.where(keep, old, jnp.where(start, new, blank)), continuing, started, empty
        )
        telemetry = telemetry | dict(
            commitment_started=start,
            commitment_continued=keep & ~arrived,
            commitment_held=hold,
            commitment_released=prior_active & ~keep,
            commitment_observation_inconsistent=prior_active & ~arrival_consistent,
            commitment_expired=prior_active & expired,
            commitment_observation_gap=prior_active & observation_gap,
            commitment_reverse_blocked=reverse,
            commitment_override=jnp.any(chosen != action),
            commitment_remaining=jnp.where(keep, remaining, 0),
            commitment_required=jnp.where(keep, required, 0),
            commitment_arrival_army=jnp.where(keep, arrival.reshape(-1)[defender], 0),
        )
        telemetry["building"] = chosen[0] == 2
        return chosen, memory, telemetry

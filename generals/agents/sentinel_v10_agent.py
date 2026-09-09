"""Experimental home-army mobilization after deathtouch becomes active.

Compare a real home-sourced route with the parent's newly selected field
interception at the same target. This changes source eligibility, not the
parent's approximate route certificate. It does not repair stale donation
accounting, predict merges, or guarantee safety against adaptive detours.

Inherited telemetry describes the parent proposal. Mask its issued-action
events with mobilization_parent_action_issued. Disabled operation delegates
exact actions, memory and telemetry to the selected frozen parent.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance
from .sentinel_v5_agent import _HORIZON, _owned_route_plan
from .sentinel_v6_agent import SentinelV6Agent, _destination
from .sentinel_v9_agent import SentinelV9Agent


class SentinelV10Agent:
    def __init__(
        self, id="Sentinel-v10", *, build_castles=False, deathtouch_turn=None,
        max_turns=1200, parent_version=9, mobilize_home=True,
    ):
        if parent_version not in (6, 9):
            raise ValueError("parent_version must be 6 or 9")
        self.id = id
        self.deathtouch_turn = deathtouch_turn
        self.parent_version = parent_version
        self.mobilize_home = mobilize_home
        cls = SentinelV6Agent if parent_version == 6 else SentinelV9Agent
        self.base = cls(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        return self.base.initial_memory(shape)

    def _defense(self, memory):
        return memory if self.parent_version == 6 else memory.base.defense

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.base.step(obs, key, memory)
        if not self.mobilize_home:
            return action, returned, telemetry

        a, mine = obs.armies, obs.owned_cells
        h, w = a.shape
        grid = jnp.arange(a.size).reshape(a.shape)
        home = mine & obs.generals
        home_index = jnp.argmax(home)
        target_index = telemetry["intercept_target_index"]
        enemy_index = telemetry["intercept_threat_index"]
        target = grid == target_index
        enemy = grid == enemy_index
        terrain = ~(obs.mountains | obs.structures_in_fog)
        time = obs.timestep.astype(jnp.int32)
        active = time >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        prior, defense = self._defense(memory), self._defense(returned)
        dest = _destination(action, w, a.shape)
        eligible = (
            active & jnp.any(home) & (jnp.sum(jnp.where(home, a, 0)) > 1)
            & (prior.defender < 0) & telemetry["commitment_started"]
            & telemetry["intercept_feasible"] & (telemetry["intercept_home_deficit"] > 0)
            & ~telemetry["intercept_guard"] & (telemetry["adjacent_threat"] <= 0)
            & (action[0] == 0) & (defense.defender == dest)
            & (target_index >= 0) & (target_index < a.size)
            & (enemy_index >= 0) & (enemy_index < a.size)
        )

        def propose(_):
            enemy_distance = _distance(terrain, enemy)
            costs = 1 + jnp.where(mine | obs.neutral_cells, a, 0)
            home_cost = _distance(terrain, home, costs) - jnp.sum(jnp.where(home, a, 0))
            enemy_cost = _distance(terrain, enemy, costs) + costs - costs.reshape(-1)[enemy_index]
            corridor = enemy_cost + home_cost <= home_cost.reshape(-1)[enemy_index]
            # The general is a real donor after deathtouch: its stationary army
            # does not resist capture. No other garrison eligibility changes.
            distance, extra, arrival, deadline, direction = _owned_route_plan(
                obs, terrain, jnp.zeros_like(home), enemy_distance, corridor & ~home, target
            )
            eta = distance.reshape(-1)[home_index]
            force = arrival.reshape(-1)[home_index]
            donation = extra.reshape(-1)[home_index]
            move = jnp.array([0, home_index // w, home_index % w, direction.reshape(-1)[home_index], 0], jnp.int32)
            destination = _destination(move, w, a.shape)
            valid = (
                (eta >= 1) & (eta <= _HORIZON)
                & (eta <= deadline.reshape(-1)[home_index])
                & (eta <= enemy_distance.reshape(-1)[target_index])
                & (eta <= telemetry["intercept_friendly_eta"])
                & (force >= telemetry["intercept_arrival_army"])
                & (donation >= telemetry["intercept_required_defense"])
                & ((eta < telemetry["intercept_friendly_eta"]) | (force > telemetry["intercept_arrival_army"]))
                & mine.reshape(-1)[destination]
            )
            # Recompute actual source depletion. Subtract each position's own
            # home garrison; subtracting the old home army twice would invent
            # lost resistance when those armies were already ineffective.
            after = a.reshape(-1).at[home_index].set(1)
            after = after.at[destination].add(a.reshape(-1)[home_index] - 1).reshape(a.shape)
            after_cost = _distance(terrain, home, 1 + jnp.where(mine | obs.neutral_cells, after, 0))
            after_cost -= jnp.sum(jnp.where(home, after, 0))
            relevant = obs.opponent_cells & (_distance(terrain, home) <= _HORIZON)
            preserves = jnp.all(~relevant | jnp.where(a <= home_cost, a <= after_cost, after_cost >= home_cost))
            return move, valid & preserves, eta, force, donation, preserves

        proposed, use, eta, force, donation, safe = jax.lax.cond(
            eligible, propose,
            lambda _: (action, jnp.array(False), jnp.float32(0), jnp.float32(0), jnp.float32(0), jnp.array(False)),
            operand=None,
        )
        changed = use & jnp.any(proposed != action)
        blank = self.initial_memory(a.shape)
        if self.parent_version == 6:
            cleared = blank._replace(last_turn=time)
        else:
            cleared_defense = blank.base.defense._replace(last_turn=time)
            cleared = returned._replace(base=blank.base._replace(defense=cleared_defense, last_turn=time))
        returned = jax.tree.map(lambda old, new: jnp.where(changed, new, old), returned, cleared)
        return jnp.where(changed, proposed, action), returned, telemetry | dict(
            mobilization_eligible=eligible,
            mobilization_issued=changed,
            mobilization_parent_action_issued=~changed,
            mobilization_home_eta=eta,
            mobilization_arrival_army=force,
            mobilization_donation=donation,
            mobilization_home_coverage_safe=safe,
        )

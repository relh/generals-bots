"""Experimental bounded continuation of an existing owned defender.

Loss of current visible need may release V6's commitment before its owned
screen has finished serving its original plan. Retention here remembers only
that existing defender, target and expiry. It does not remember enemy forces,
refresh a deadline, or certify defense against hidden or adaptive attacks.

The wrapper sits at the actual V6 decision inside V18. Standard commitment
telemetry describes the resulting inner decision; retained_parent_* preserves
the original V6 proposal. retained_action_issued additionally accounts for
all outer action selectors. Disabled operation is the exact V18 tuple.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance, _neighbors
from .sentinel_v3_agent import _campaign_decision
from .sentinel_v5_agent import _HORIZON, _home_growth
from .sentinel_v6_agent import DefenderMemory, _destination
from .sentinel_v18_agent import SentinelV18Agent

_INF = 1e6
_ACTION_FIELDS = ("kind", "row", "column", "direction", "split")


class _RetainedDefender:
    """One original V6 step followed by arbitration over its actual result."""

    def __init__(self, parent):
        self.parent = parent
        self.deathtouch_turn = parent.deathtouch_turn

    def initial_memory(self, shape):
        return self.parent.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.parent.step(obs, key, memory)
        armies, mine = obs.armies, obs.owned_cells
        width = armies.shape[1]
        cells = jnp.arange(armies.size).reshape(armies.shape)
        time = obs.timestep.astype(jnp.int32)
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        defender = jnp.clip(memory.defender, 0, armies.size - 1)
        target_index = jnp.clip(memory.target, 0, armies.size - 1)
        target = cells == target_index
        defender_army = armies.reshape(-1)[defender]
        indices_valid = ((memory.defender >= 0) & (memory.defender < armies.size)
                         & (memory.target >= 0) & (memory.target < armies.size))
        consecutive = memory.last_turn + 1 == time
        unexpired = time <= memory.expires
        consistent = (mine.reshape(-1)[defender] & mine.reshape(-1)[target_index]
                      & ~home.reshape(-1)[defender] & (defender_army > 1)
                      & (defender_army >= memory.expected_army))
        valid = indices_valid & consecutive & unexpired & consistent & jnp.any(home)

        # Reconstruct V6's current-visible need with this owned defender
        # removed from the route cost. No past enemy observation is consulted.
        home_distance = _distance(terrain, home)
        without = armies.reshape(-1).at[defender].set(1).reshape(armies.shape)
        without_cost = _distance(terrain, home, 1 + jnp.where(mine | obs.neutral_cells, without, 0))
        projected_touch = (time + home_distance - 1 >= self.deathtouch_turn
                           if self.deathtouch_turn is not None else jnp.zeros_like(mine))
        without_cost += jnp.where(projected_touch, -jnp.sum(jnp.where(home, armies, 0)),
                                  _home_growth(time, home_distance))
        visible_need = jnp.any(obs.opponent_cells & (home_distance <= _HORIZON)
                               & (armies > without_cost))

        parent_source = action[1] * width + action[2]
        parent_target = _destination(action, width, armies.shape)
        parent_moved = jnp.where(action[4] == 1, armies.reshape(-1)[parent_source] // 2,
                                 armies.reshape(-1)[parent_source] - 1)
        touch_active = (time >= self.deathtouch_turn if self.deathtouch_turn is not None
                        else jnp.array(False))
        winning_general = ((action[0] == 0)
                           & (obs.opponent_cells & obs.generals).reshape(-1)[parent_target]
                           & ((parent_moved > armies.reshape(-1)[parent_target]) | touch_active))
        selected_nonowned_move = (action[0] == 0) & ~mine.reshape(-1)[parent_target]
        selected_build = action[0] == 2
        priority = ((telemetry["adjacent_threat"] > 0) | telemetry["intercept_guard"]
                    | winning_general | selected_nonowned_move | selected_build)
        parent_released = (returned.defender < 0) & ~telemetry["commitment_started"]

        # Existing shortest owned route only. Prefer the original full move
        # when it progresses; otherwise use the first U/D/L/R shortest step.
        route_cells = terrain & mine & (~home | target)
        distance = _distance(route_cells, target)
        remaining = distance.reshape(-1)[defender]
        arrived = defender == target_index
        next_distance = _neighbors(distance, _INF).reshape(-1, 4)[defender]
        next_owned = _neighbors(route_cells, False).reshape(-1, 4)[defender]
        directions = next_owned & (next_distance == remaining - 1)
        route_valid = ((remaining > 0) & (remaining < _INF / 2)
                       & (remaining <= memory.distance) & jnp.any(directions))
        parent_progresses = ((action[0] == 0) & (action[4] == 0)
                             & (parent_source == defender) & mine.reshape(-1)[parent_target]
                             & (distance.reshape(-1)[parent_target] == remaining - 1))
        direction = jnp.argmax(directions)
        first_step = jnp.array([0, defender // width, defender % width, direction, 0], jnp.int32)
        transit = jnp.where(parent_progresses, action, first_step)
        eligible = valid & parent_released & ~visible_need & ~priority & (arrived | route_valid)
        parent_uses_defender = (action[0] == 0) & (parent_source == defender)

        def alternative(_):
            # Frozen campaign scores and tie order, excluding only moves from
            # the held defender. Already selected builds/nonowned moves gate us out.
            _, _, scores = _campaign_decision(self.parent.base.campaign, obs, key)
            scores = jnp.where((cells == defender)[..., None, None], -1e9, scores)
            best = jnp.argmax(scores)
            source = best // 8
            return jnp.array([jnp.max(scores) <= 0, source // width, source % width,
                              (best // 2) % 4, best % 2], jnp.int32)

        hold_action = jax.lax.cond(eligible & arrived & parent_uses_defender, alternative,
                                   lambda _: action, operand=None)
        proposed = jnp.where(arrived, hold_action, transit)
        chosen = jnp.where(eligible, proposed, action)
        destination = _destination(chosen, width, armies.shape)
        finished = eligible & ~arrived & home.reshape(-1)[destination]
        expected = jnp.where(arrived, defender_army,
                             defender_army - 1 + armies.reshape(-1)[destination])
        retained = DefenderMemory(
            jnp.where(arrived, defender, destination).astype(jnp.int32), memory.target,
            jnp.where(arrived, memory.previous, defender).astype(jnp.int32),
            jnp.where(arrived, 0, remaining - 1).astype(jnp.int32), memory.expires,
            time, expected.astype(jnp.int32),
        )
        # Arrival at home finishes the plan immediately; home is not a reusable
        # field defender. No continuation, including a hold, extends expiry.
        empty = self.initial_memory(armies.shape)._replace(last_turn=time)
        retained = jax.tree.map(lambda value, blank: jnp.where(finished, blank, value), retained, empty)
        result_memory = jax.tree.map(lambda value, original: jnp.where(eligible, value, original), retained, returned)
        changed = eligible & jnp.any(chosen != action)
        updated = dict(
            commitment_started=jnp.array(False),
            commitment_continued=~arrived,
            commitment_held=arrived,
            commitment_released=finished,
            commitment_observation_inconsistent=jnp.array(False),
            commitment_expired=jnp.array(False),
            commitment_observation_gap=jnp.array(False),
            commitment_reverse_blocked=jnp.array(False),
            commitment_override=changed,
            commitment_remaining=remaining,
            commitment_required=jnp.float32(0),
        )
        actual_telemetry = telemetry | {name: jnp.where(eligible, value, telemetry[name])
                                        for name, value in updated.items()}
        actual_telemetry["building"] = jnp.where(eligible, chosen[0] == 2, telemetry["building"])
        return chosen, result_memory, actual_telemetry | {
            **{"retained_parent_" + name: action[i] for i, name in enumerate(_ACTION_FIELDS)},
            **{"retained_parent_" + name: value for name, value in telemetry.items()},
            **{"retained_proposal_" + name: proposed[i] for i, name in enumerate(_ACTION_FIELDS)},
            "retained_indices_valid": indices_valid,
            "retained_consecutive": consecutive,
            "retained_unexpired": unexpired,
            "retained_consistent": consistent,
            "retained_valid": valid,
            "retained_parent_released": parent_released,
            "retained_visible_need": visible_need,
            "retained_winning_general": winning_general,
            "retained_selected_nonowned_move": selected_nonowned_move,
            "retained_selected_build": selected_build,
            "retained_priority": priority,
            "retained_arrived": arrived & indices_valid,
            "retained_route_valid": route_valid,
            "retained_remaining": jnp.where(indices_valid, remaining, -1),
            "retained_parent_progresses": parent_progresses,
            "retained_eligible": eligible,
            "retained_continued": eligible & ~arrived,
            "retained_held": eligible & arrived,
            "retained_finished": finished,
            "retained_source": jnp.where(eligible, defender, -1),
            "retained_target": jnp.where(eligible, target_index, -1),
            "retained_expires": memory.expires,
            "retained_expected_army": jnp.where(eligible, expected, -1),
            "retained_inner_issued": eligible,
            "retained_inner_override": changed,
            "retained_parent_action_issued": ~changed,
        }


class SentinelV19Agent:
    """V18 with bounded retention at its actual inner V6 decision layer."""

    def __init__(self, id="Sentinel-v19", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, retain_defense=True):
        self.id = id
        self.retain_defense = retain_defense
        self.parent = SentinelV18Agent(build_castles=build_castles,
                                       deathtouch_turn=deathtouch_turn, max_turns=max_turns)
        if retain_defense:
            inner = self.parent.parent.base.base
            inner.base = _RetainedDefender(inner.base)

    def initial_memory(self, shape):
        return self.parent.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.parent.step(obs, key, memory)
        if not self.retain_defense:
            return action, returned, telemetry
        outer_issued = (telemetry["rear_parent_action_issued"]
                        & telemetry["mobilization_parent_action_issued"]
                        & telemetry["actual_v8_action_issued"] & ~telemetry["offense_override"])
        return action, returned, telemetry | {
            "retained_outer_action_issued": outer_issued,
            "retained_action_issued": telemetry["retained_inner_issued"] & outer_issued,
            "retained_action_override": telemetry["retained_inner_override"] & outer_issued,
        }

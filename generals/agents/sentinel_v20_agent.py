"""Experimental budgeted branching collection over the frozen V10 commander.

A shortest-owned tree can combine side branches that no single collection path
can include. Every gathered edge consumes one actual action; current observations
replace donor estimates after each move. This is bounded offensive planning, not
an adaptive safety certificate or a maintained reserve.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .branch_collection import plan_branch_collection
from .sentinel_agent import _distance, _neighbors
from .sentinel_v6_agent import _destination
from .sentinel_v7_agent import _collection_route, _preserves_home
from .sentinel_v9_agent import StrategicMemory
from .sentinel_v10_agent import SentinelV10Agent

_INF = 1e6
_BUDGET = 6


class BranchMemory(NamedTuple):
    parent: StrategicMemory
    objective: jax.Array
    rally: jax.Array
    packet: jax.Array
    phase: jax.Array
    budget: jax.Array
    expires: jax.Array
    last_turn: jax.Array
    expected_army: jax.Array
    remaining: jax.Array


class SentinelV20Agent:
    def __init__(self, id="Sentinel-v20", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, branch_collection=True):
        self.id = id
        self.branch_collection = branch_collection
        self.deathtouch_turn = deathtouch_turn
        self.parent = SentinelV10Agent(build_castles=build_castles,
                                       deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def _empty(self, parent, time):
        return BranchMemory(parent, *(jnp.int32(x) for x in (-1, -1, -1, 0, 0, -1)),
                            jnp.asarray(time, jnp.int32), jnp.int32(0), jnp.int32(0))

    def initial_memory(self, shape):
        parent = self.parent.initial_memory(shape)
        return self._empty(parent, -1) if self.branch_collection else parent

    def _telemetry(self):
        return dict(branch_available=jnp.array(False), branch_started=jnp.array(False),
                    branch_continued=jnp.array(False), branch_collecting=jnp.array(False),
                    branch_deploying=jnp.array(False), branch_attack_issued=jnp.array(False),
                    branch_home_coverage_safe=jnp.array(False),
                    branch_objective=jnp.int32(-1), branch_rally=jnp.int32(-1),
                    branch_selected_budget=jnp.int32(-1), branch_delivered=jnp.float32(0),
                    branch_required=jnp.float32(0), branch_single_path_sufficient=jnp.array(False))

    def _propose(self, obs, memory, parent_action, returned):
        a, mine = obs.armies, obs.owned_cells
        h, w = a.shape
        flat = a.reshape(-1)
        cells = jnp.arange(a.size).reshape(a.shape)
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        local_force = jnp.max(_neighbors(jnp.where(obs.opponent_cells, jnp.maximum(a - 1, 0), 0), 0), axis=-1)
        eligible = mine & ~home & (local_force == 0)
        surplus = jnp.where(eligible, jnp.maximum(a - 1, 0), 0)
        pool = jax.lax.reduce_window(surplus, 0, jax.lax.add, (7, 7), (1, 1), "SAME")
        touch_now = (obs.timestep >= self.deathtouch_turn
                     if self.deathtouch_turn is not None else jnp.array(False))
        target_force = jnp.where(obs.generals & touch_now, 0, a)
        targets = obs.opponent_cells & terrain & (pool > target_force + 2)
        target_score = jnp.where(targets, a + obs.castles * 20 + obs.generals * 100 - home_distance * .1, -_INF)
        had = memory.phase > 0
        objective = jnp.clip(jnp.where(had, memory.objective, jnp.argmax(target_score)), 0, a.size - 1)
        target = cells == objective
        attack_distance = _distance(terrain & (mine | target) & ~home, target)
        stages = eligible & (attack_distance >= 2) & (attack_distance <= 3) & (home_distance >= 3)
        rally_score = jnp.where(stages, a + pool * .15 - attack_distance * 2, -_INF)
        rally = jnp.clip(jnp.where(had, memory.rally, jnp.argmax(rally_score)), 0, a.size - 1)
        attack_eta = attack_distance.reshape(-1)[rally]
        counter = local_force.reshape(-1)[objective]
        general = obs.generals.reshape(-1)[objective]
        budgets = jnp.arange(_BUDGET + 1)
        projected_touch = (obs.timestep + budgets + attack_eta - 1 >= self.deathtouch_turn
                           if self.deathtouch_turn is not None else jnp.zeros_like(budgets, dtype=bool))
        required = jnp.where(general & projected_touch, attack_eta + 1,
                             flat[objective] + counter + attack_eta + 1)
        # The same original V8 target/rally and six-step path comparator prevent
        # describing an ordinary serial route as new branch-only availability.
        path_distance, path_delivered, _, _ = _collection_route(a, eligible, cells == rally)
        path_touch = (obs.timestep + path_distance + attack_eta - 1 >= self.deathtouch_turn
                      if self.deathtouch_turn is not None else jnp.zeros_like(mine))
        path_required = jnp.where(general & path_touch, attack_eta + 1,
                                  flat[objective] + counter + attack_eta + 1)
        path_ok = (eligible & (path_distance >= 1) & (path_distance <= _BUDGET)
                   & (path_delivered >= path_required))
        tree = plan_branch_collection(a, eligible, rally)
        limit = jnp.where(had, memory.budget, _BUDGET)
        sufficient = (tree.feasible & (budgets <= limit) & (tree.delivered >= required)
                      & (had | ((budgets >= 2) & (tree.delivered >= tree.largest_garrison * 1.5))))
        selected = jnp.argmin(jnp.where(sufficient, budgets, _BUDGET + 1))
        available = jnp.any(sufficient)
        gather = (memory.phase != 2) & (selected > 0)
        packet = jnp.clip(jnp.where(memory.phase == 2, memory.packet, rally), 0, a.size - 1)
        deploy_eta = attack_distance.reshape(-1)[packet]
        deploy_direction = jnp.argmin(_neighbors(attack_distance, _INF), axis=-1).reshape(-1)[packet]
        source = jnp.clip(jnp.where(gather, tree.first_source[selected], packet), 0, a.size - 1)
        direction = jnp.where(gather, tree.first_direction[selected], deploy_direction)
        proposed = jnp.array([0, source // w, source % w, direction, 0], jnp.int32)
        destination = _destination(proposed, w, a.shape)
        destination_owned = mine.reshape(-1)[destination]
        deploy_touch = (obs.timestep + deploy_eta - 1 >= self.deathtouch_turn
                        if self.deathtouch_turn is not None else jnp.array(False))
        deploy_need = jnp.where(general & deploy_touch, deploy_eta + 1,
                                flat[objective] + counter + deploy_eta + 1)
        enough = jnp.where(gather, available, flat[packet] >= deploy_need)
        # Once deploying, this packet alone must suffice. No tree's current or
        # remembered donors are credited to a packet that has left the rally.
        remaining = jnp.where(gather, selected, deploy_eta)
        destination_ok = destination_owned | (destination == objective)
        captures = (flat[source] - 1 > flat[destination]) | (general & touch_now & (destination == objective))
        legal = (mine.reshape(-1)[source] & ~home.reshape(-1)[source] & (flat[source] > 1)
                 & destination_ok & (destination_owned | captures) & (remaining >= 1)
                 & (remaining < _INF / 2))
        legal &= jnp.where(gather, eligible.reshape(-1)[source] & eligible.reshape(-1)[destination], True)
        start = (~had & jnp.any(targets) & jnp.any(stages) & available & ~jnp.any(path_ok))
        last_packet = jnp.clip(memory.packet, 0, a.size - 1)
        consistent = (mine.reshape(-1)[last_packet] & (flat[last_packet] >= memory.expected_army))
        continuation = (had & (memory.last_turn + 1 == obs.timestep) & (obs.timestep <= memory.expires)
                        & consistent & jnp.where(memory.phase == 2, deploy_eta <= memory.remaining,
                                                eligible.reshape(-1)[rally] & (attack_eta >= 1) & (attack_eta <= 3)))
        # During gathering the current rally must still have a public owned
        # deployment route. During deployment only the actual packet is needed.
        remaining_work = jnp.where(gather, selected + attack_eta, deploy_eta)
        continuation &= obs.timestep + remaining_work - 1 <= memory.expires
        objective_valid = obs.opponent_cells.reshape(-1)[objective]
        safe = _preserves_home(obs, proposed, terrain, home, home_distance, self.deathtouch_turn)
        use = (start | continuation) & objective_valid & legal & enough & safe
        attacking = use & (destination == objective)
        next_phase = jnp.where(gather & (selected > 1), 1, 2)
        next_remaining = jnp.where(gather, attack_eta, deploy_eta - 1)
        chosen_parent = returned
        # Drop only unissued offensive transport; retain the parent's real
        # defender and public stationary-general knowledge.
        blank_offense = self.parent.initial_memory(a.shape).base
        chosen_parent = chosen_parent._replace(base=blank_offense._replace(
            defense=returned.base.defense, last_turn=obs.timestep.astype(jnp.int32)))
        next_memory = BranchMemory(
            chosen_parent, objective.astype(jnp.int32), rally.astype(jnp.int32), destination.astype(jnp.int32),
            next_phase.astype(jnp.int32), jnp.where(gather, selected - 1, 0).astype(jnp.int32),
            jnp.where(had, memory.expires, obs.timestep + selected + attack_eta + 1).astype(jnp.int32),
            obs.timestep.astype(jnp.int32), (flat[source] - 1 + flat[destination]).astype(jnp.int32),
            next_remaining.astype(jnp.int32))
        empty = self._empty(returned, obs.timestep)
        # A completed capture clears transport, but the actual parent's
        # unissued proposal must still be cleared even on this final move.
        cleared = self._empty(chosen_parent, obs.timestep)
        next_memory = jax.tree.map(lambda planned, end: jnp.where(attacking, end, planned), next_memory, cleared)
        next_memory = jax.tree.map(lambda planned, fallback: jnp.where(use, planned, fallback), next_memory, empty)
        return jnp.where(use, proposed, parent_action), next_memory, dict(
            branch_available=available, branch_started=use & ~had, branch_continued=use & had,
            branch_collecting=use & gather, branch_deploying=use & ~gather, branch_attack_issued=attacking,
            branch_home_coverage_safe=safe, branch_objective=jnp.where(use, objective, -1).astype(jnp.int32),
            branch_rally=jnp.where(use, rally, -1).astype(jnp.int32),
            branch_selected_budget=jnp.where(use & gather, selected, -1).astype(jnp.int32),
            branch_delivered=jnp.where(use & gather, tree.delivered[selected], 0).astype(jnp.float32),
            branch_required=jnp.where(gather, required[selected], deploy_need).astype(jnp.float32),
            branch_single_path_sufficient=jnp.any(path_ok))

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        if not self.branch_collection:
            return self.parent.step(obs, key, memory)
        action, returned, telemetry = self.parent.step(obs, key, memory.parent)
        time = obs.timestep.astype(jnp.int32)
        shape_valid = (memory.parent.height == obs.armies.shape[0]) & (memory.parent.width == obs.armies.shape[1])
        prior_valid = shape_valid & (memory.last_turn + 1 == time) & (time > memory.parent.last_turn)
        # Invalid incoming transport cannot constrain a new game's first move.
        effective = jax.tree.map(lambda old, blank: jnp.where(prior_valid, old, blank),
                                 memory, self._empty(self.parent.initial_memory(obs.armies.shape), time - 1))
        offense = (telemetry['actual_v8_action_issued'] & telemetry['mobilization_parent_action_issued']
                   & (telemetry['offense_started'] | telemetry['offense_continued']))
        priority = ((action[0] == 2) | (effective.parent.base.defense.defender >= 0)
                    | (returned.base.defense.defender >= 0) | telemetry['intercept_guard']
                    | (telemetry['intercept_home_deficit'] > 0) | (telemetry['adjacent_threat'] > 0)
                    | (telemetry['general_reserve'] > telemetry['general_army']) | offense
                    | telemetry['pursuit_issued'] | telemetry['mobilization_issued'])
        # Existing parent transport also keeps priority if its current step is
        # blocked; branch collection never claims its incoming commitment.
        priority |= prior_valid & (memory.parent.base.phase > 0)
        dest = _destination(action, obs.armies.shape[1], obs.armies.shape)
        source = action[1] * obs.armies.shape[1] + action[2]
        moving = jnp.where(action[4] == 1, obs.armies.reshape(-1)[source] // 2,
                           obs.armies.reshape(-1)[source] - 1)
        touch = time >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        priority |= ((action[0] == 0) & (obs.opponent_cells & obs.generals).reshape(-1)[dest]
                     & ((moving > obs.armies.reshape(-1)[dest]) | touch))
        inspect = ~priority & jnp.any(obs.opponent_cells) & jnp.any(obs.owned_cells & obs.generals)
        chosen, result, extra = jax.lax.cond(
            inspect, lambda _: self._propose(obs, effective, action, returned),
            lambda _: (action, self._empty(returned, time), self._telemetry()), operand=None)
        issued = extra['branch_started'] | extra['branch_continued']
        return chosen, result, telemetry | extra | dict(
            branch_action_issued=issued, branch_override=issued & jnp.any(chosen != action),
            branch_parent_action_issued=~issued, branch_priority=priority,
            branch_released=(effective.phase > 0) & ~extra['branch_continued'], building=chosen[0] == 2)

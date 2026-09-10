"""Experimental persistent tree collection inside the V9/V10 offensive stack.

The frozen ordinary collector runs once. An active, feasible tree competes with
new serial collection, while ready attacks and urgent safety retain priority.
This is a bounded collection experiment, not a competitive-strength claim.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .branch_collection import plan_branch_collection
from .sentinel_agent import _distance, _neighbors
from .sentinel_v6_agent import DefenderMemory, _destination
from .sentinel_v7_agent import OffensiveMemory, _collection_route, _preserves_home
from .sentinel_v8_agent import SentinelV8Agent
from .sentinel_v10_agent import SentinelV10Agent

_INF = 1e6
_BUDGET = 6


class TreeState(NamedTuple):
    objective: jax.Array
    rally: jax.Array
    packet: jax.Array
    phase: jax.Array
    budget: jax.Array
    expires: jax.Array
    last_turn: jax.Array
    expected_army: jax.Array
    remaining: jax.Array


class TreeOffensiveMemory(NamedTuple):
    defense: DefenderMemory
    packet: jax.Array
    rally: jax.Array
    objective: jax.Array
    phase: jax.Array
    remaining: jax.Array
    expires: jax.Array
    last_turn: jax.Array
    expected_army: jax.Array
    tree: TreeState


class TreeCollector:
    def __init__(self, *, build_castles=False, deathtouch_turn=None, max_turns=1200):
        self.deathtouch_turn = deathtouch_turn
        self.parent = SentinelV8Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def _empty(self, time):
        return TreeState(
            *(jnp.int32(x) for x in (-1, -1, -1, 0, 0, -1)), jnp.asarray(time, jnp.int32), jnp.int32(0), jnp.int32(0)
        )

    def initial_memory(self, shape):
        return TreeOffensiveMemory(*self.parent.initial_memory(shape), self._empty(-1))

    def _telemetry(self):
        return dict(
            tree_available=jnp.array(False),
            tree_started=jnp.array(False),
            tree_continued=jnp.array(False),
            tree_collecting=jnp.array(False),
            tree_deploying=jnp.array(False),
            tree_attack_issued=jnp.array(False),
            tree_home_coverage_safe=jnp.array(False),
            tree_objective=jnp.int32(-1),
            tree_rally=jnp.int32(-1),
            tree_selected_budget=jnp.int32(-1),
            tree_delivered=jnp.float32(0),
            tree_required=jnp.float32(0),
            tree_single_path_sufficient=jnp.array(False),
            tree_remaining_actions=jnp.int32(-1),
        )

    def _propose(self, obs, memory, parent_action):
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
        touch_now = obs.timestep >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        target_force = jnp.where(obs.generals & touch_now, 0, a)
        targets = obs.opponent_cells & terrain & (pool > target_force + 2)
        target_score = jnp.where(targets, a + obs.castles * 20 + obs.generals * 100 - home_distance * 0.1, -_INF)
        had = memory.phase > 0
        objective = jnp.clip(jnp.where(had, memory.objective, jnp.argmax(target_score)), 0, a.size - 1)
        target = cells == objective
        attack_distance = _distance(terrain & (mine | target) & ~home, target)
        stages = eligible & (attack_distance >= 2) & (attack_distance <= 3) & (home_distance >= 3)
        rally_score = jnp.where(stages, a + pool * 0.15 - attack_distance * 2, -_INF)
        rally = jnp.clip(jnp.where(had, memory.rally, jnp.argmax(rally_score)), 0, a.size - 1)
        attack_eta = attack_distance.reshape(-1)[rally]
        counter = local_force.reshape(-1)[objective]
        general = obs.generals.reshape(-1)[objective]
        budgets = jnp.arange(_BUDGET + 1)
        projected_touch = (
            obs.timestep + budgets + attack_eta - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(budgets, dtype=bool)
        )
        required = jnp.where(general & projected_touch, attack_eta + 1, flat[objective] + counter + attack_eta + 1)
        # The same original V8 target/rally and six-step path comparator prevent
        # describing an ordinary serial route as new branch-only availability.
        path_distance, path_delivered, _, _ = _collection_route(a, eligible, cells == rally)
        path_touch = (
            obs.timestep + path_distance + attack_eta - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.zeros_like(mine)
        )
        path_required = jnp.where(general & path_touch, attack_eta + 1, flat[objective] + counter + attack_eta + 1)
        path_ok = eligible & (path_distance >= 1) & (path_distance <= _BUDGET) & (path_delivered >= path_required)
        tree = plan_branch_collection(a, eligible, rally)
        limit = jnp.where(had, memory.budget, _BUDGET)
        sufficient = (
            tree.feasible
            & (budgets <= limit)
            & (tree.delivered >= required)
            & (had | ((budgets >= 2) & (tree.delivered >= tree.largest_garrison * 1.5)))
        )
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
        deploy_touch = (
            obs.timestep + deploy_eta - 1 >= self.deathtouch_turn
            if self.deathtouch_turn is not None
            else jnp.array(False)
        )
        deploy_need = jnp.where(general & deploy_touch, deploy_eta + 1, flat[objective] + counter + deploy_eta + 1)
        enough = jnp.where(gather, available, flat[packet] >= deploy_need)
        # Once deploying, this packet alone must suffice. No tree's current or
        # remembered donors are credited to a packet that has left the rally.
        remaining = jnp.where(gather, selected, deploy_eta)
        destination_ok = destination_owned | (destination == objective)
        captures = (flat[source] - 1 > flat[destination]) | (general & touch_now & (destination == objective))
        legal = (
            mine.reshape(-1)[source]
            & ~home.reshape(-1)[source]
            & (flat[source] > 1)
            & destination_ok
            & (destination_owned | captures)
            & (remaining >= 1)
            & (remaining < _INF / 2)
        )
        legal &= jnp.where(gather, eligible.reshape(-1)[source] & eligible.reshape(-1)[destination], True)
        start = ~had & jnp.any(targets) & jnp.any(stages) & available & ~jnp.any(path_ok)
        last_packet = jnp.clip(memory.packet, 0, a.size - 1)
        consistent = mine.reshape(-1)[last_packet] & (flat[last_packet] >= memory.expected_army)
        continuation = (
            had
            & (memory.last_turn + 1 == obs.timestep)
            & (obs.timestep <= memory.expires)
            & consistent
            & jnp.where(
                memory.phase == 2,
                deploy_eta <= memory.remaining,
                eligible.reshape(-1)[rally] & (attack_eta >= 1) & (attack_eta <= 3),
            )
        )
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
        next_memory = TreeState(
            objective.astype(jnp.int32),
            rally.astype(jnp.int32),
            destination.astype(jnp.int32),
            next_phase.astype(jnp.int32),
            jnp.where(gather, selected - 1, 0).astype(jnp.int32),
            jnp.where(had, memory.expires, obs.timestep + selected + attack_eta + 1).astype(jnp.int32),
            obs.timestep.astype(jnp.int32),
            (flat[source] - 1 + flat[destination]).astype(jnp.int32),
            next_remaining.astype(jnp.int32),
        )
        empty = self._empty(obs.timestep)
        cleared = empty
        next_memory = jax.tree.map(lambda planned, end: jnp.where(attacking, end, planned), next_memory, cleared)
        next_memory = jax.tree.map(lambda planned, fallback: jnp.where(use, planned, fallback), next_memory, empty)
        return (
            jnp.where(use, proposed, parent_action),
            next_memory,
            dict(
                tree_available=available,
                tree_started=use & ~had,
                tree_continued=use & had,
                tree_collecting=use & gather,
                tree_deploying=use & ~gather,
                tree_attack_issued=attacking,
                tree_home_coverage_safe=safe,
                tree_objective=jnp.where(use, objective, -1).astype(jnp.int32),
                tree_rally=jnp.where(use, rally, -1).astype(jnp.int32),
                tree_selected_budget=jnp.where(use & gather, selected, -1).astype(jnp.int32),
                tree_delivered=jnp.where(use & gather, tree.delivered[selected], 0).astype(jnp.float32),
                tree_required=jnp.where(gather, required[selected], deploy_need).astype(jnp.float32),
                tree_single_path_sufficient=jnp.any(path_ok),
                tree_remaining_actions=remaining_work.astype(jnp.int32),
            ),
        )

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        ordinary = OffensiveMemory(*(getattr(memory, name) for name in OffensiveMemory._fields))
        action, returned, telemetry = self.parent.step(obs, key, ordinary)
        time = obs.timestep.astype(jnp.int32)
        tree = memory.tree
        had = tree.phase > 0
        offense = telemetry["offense_started"] | telemetry["offense_continued"]
        ready = offense & telemetry["offense_deploying"]
        urgent = (
            (action[0] == 2)
            | (ordinary.defense.defender >= 0)
            | (returned.defense.defender >= 0)
            | telemetry["intercept_guard"]
            | (telemetry["intercept_home_deficit"] > 0)
            | (telemetry["adjacent_threat"] > 0)
            | (telemetry["general_reserve"] > telemetry["general_army"])
        )
        destination = _destination(action, obs.armies.shape[1], obs.armies.shape)
        source = action[1] * obs.armies.shape[1] + action[2]
        flat = obs.armies.reshape(-1)
        moving = jnp.where(action[4] == 1, flat[source] // 2, flat[source] - 1)
        touch = time >= self.deathtouch_turn if self.deathtouch_turn is not None else jnp.array(False)
        urgent |= (
            (action[0] == 0)
            & (obs.opponent_cells & obs.generals).reshape(-1)[destination]
            & ((moving > flat[destination]) | touch)
        )
        # The tree owns a distinct phase. Existing ordinary memory is never
        # commandeered, including an ordinary plan whose current move is blocked.
        priority = urgent | ready | (ordinary.phase > 0) | (~had & offense)
        inspect = ~priority & jnp.any(obs.opponent_cells) & jnp.any(obs.owned_cells & obs.generals)
        proposed, next_tree, extra = jax.lax.cond(
            inspect,
            lambda _: self._propose(obs, tree, action),
            lambda _: (action, self._empty(time), self._telemetry()),
            operand=None,
        )
        feasible = extra["tree_started"] | extra["tree_continued"]
        serial_eta = telemetry["offense_collection_eta"]
        serial_steps = telemetry["offense_remaining"]
        handoff = (
            had
            & feasible
            & telemetry["offense_started"]
            & telemetry["offense_collecting"]
            & (source == tree.packet)
            & (telemetry["offense_objective"] == tree.objective)
            & (serial_steps <= tree.budget)
            & (serial_eta >= 1)
            & (serial_eta <= extra["tree_remaining_actions"])
            & (time + serial_eta - 1 <= tree.expires)
        )
        selected = feasible & ~handoff
        returned = returned._replace(
            expires=jnp.where(handoff, jnp.minimum(returned.expires, tree.expires), returned.expires)
        )
        blank = self.parent.initial_memory(obs.armies.shape)._replace(defense=returned.defense, last_turn=time)
        actual_ordinary = jax.tree.map(lambda b, p: jnp.where(selected, b, p), blank, returned)
        actual_tree = jax.tree.map(lambda t, b: jnp.where(selected, t, b), next_tree, self._empty(time))
        result = TreeOffensiveMemory(*actual_ordinary, actual_tree)
        # Proposal fields are retained for diagnosis. Count execution with
        # tree_collector_selected and the unchanged outer V9/V10 issuance masks.
        return (
            jnp.where(selected, proposed, action),
            result,
            telemetry
            | extra
            | dict(
                tree_collector_selected=selected,
                tree_parent_action_issued=~selected,
                tree_serial_handoff=handoff,
                tree_ready_handoff=had & ready & ~urgent,
                tree_urgent_priority=urgent,
                tree_priority=priority,
                tree_released=had & ~selected,
                building=jnp.where(selected, proposed[0], action[0]) == 2,
            ),
        )


class SentinelV21Agent(SentinelV10Agent):
    def __init__(
        self, id="Sentinel-v21", *, build_castles=False, deathtouch_turn=None, max_turns=1200, tree_collection=True
    ):
        super().__init__(id=id, build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)
        self.tree_collection = tree_collection
        if tree_collection:
            self.base.base = TreeCollector(
                build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns
            )

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = super().step(obs, key, memory)
        if not self.tree_collection:
            return action, returned, telemetry
        issued = (
            telemetry["tree_collector_selected"]
            & telemetry["actual_v8_action_issued"]
            & telemetry["mobilization_parent_action_issued"]
        )
        return action, returned, telemetry | dict(tree_action_issued=issued)

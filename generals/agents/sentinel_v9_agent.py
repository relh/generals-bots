"""Remember only an enemy general's previously observed stationary location.

A remembered coordinate supplies a strategic destination, never fabricated
current armies, ownership or visibility. Current observations decide each move.
The extra behavior is limited to pursuit while that coordinate is hidden;
defense, visible-general tactics and actually selected builds keep priority.

Inherited V8 telemetry describes its proposal. Consumers must mask inherited
offensive events with ``actual_v8_action_issued`` when counting issued actions.
``pursuit_issued`` includes agreement with V8; ``pursuit_override`` identifies
an actual changed output. Call initial_memory for every new game, including
same-shaped games whose first observed turn is later than the previous game.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs

from .sentinel_agent import _distance, _neighbors
from .sentinel_v7_agent import OffensiveMemory, _preserves_home
from .sentinel_v8_agent import SentinelV8Agent

_INF = 1e6


class StrategicMemory(NamedTuple):
    base: OffensiveMemory
    enemy_general: jax.Array
    last_turn: jax.Array
    height: jax.Array
    width: jax.Array


class SentinelV9Agent:
    """Disabled memory preserves exact V8 output and its full nested memory."""

    def __init__(
        self,
        id="Sentinel-v9",
        *,
        build_castles=False,
        deathtouch_turn=None,
        max_turns=1200,
        remember_enemy_general=True,
    ):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.remember_enemy_general = remember_enemy_general
        self.base = SentinelV8Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        return StrategicMemory(
            self.base.initial_memory(shape), jnp.int32(-1), jnp.int32(-1), jnp.int32(shape[0]), jnp.int32(shape[1])
        )

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        h, w = obs.armies.shape
        time = obs.timestep.astype(jnp.int32)
        if not self.remember_enemy_general:
            action, base_memory, telemetry = self.base.step(obs, key, memory.base)
            return action, StrategicMemory(base_memory, jnp.int32(-1), time, jnp.int32(h), jnp.int32(w)), telemetry

        reset = (memory.height != h) | (memory.width != w) | (time <= memory.last_turn)
        gap = (memory.last_turn >= 0) & (time > memory.last_turn + 1)
        blank_base = self.base.initial_memory(obs.armies.shape)
        # Missing observations invalidate assumed own transport. The stationary
        # destination can survive a monotonic gap in the same shaped game.
        base_input = jax.tree.map(lambda old, blank: jnp.where(reset | gap, blank, old), memory.base, blank_base)
        base_action, base_memory, telemetry = self.base.step(obs, key, base_input)
        armies, mine = obs.armies, obs.owned_cells
        home = mine & obs.generals
        seen_general = obs.opponent_cells & obs.generals
        seen = jnp.any(seen_general)
        prior = jnp.where(reset, -1, memory.enemy_general)
        cached = jnp.where(seen, jnp.argmax(seen_general), prior)
        in_range = (cached >= 0) & (cached < armies.size)
        index = jnp.clip(cached, 0, armies.size - 1)
        hidden_cells = obs.fog_cells | obs.structures_in_fog
        contradiction = in_range & ~hidden_cells.reshape(-1)[index] & ~seen_general.reshape(-1)[index]
        known = in_range & ~contradiction
        hidden = known & hidden_cells.reshape(-1)[index]
        goal_index = jnp.where(known, index, -1).astype(jnp.int32)
        invalidated = (memory.enemy_general >= 0) & (reset | ~in_range | contradiction)
        goal = (jnp.arange(armies.size).reshape(armies.shape) == index) & known

        allied = jnp.zeros_like(mine) if obs.allied_cells is None else obs.allied_cells
        friendly = mine | allied
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        affordable = obs.castles & obs.neutral_cells & (armies + 6 < jnp.max(jnp.where(mine, armies, 0)))
        # Only the remembered endpoint is known passable from history. Other
        # hidden structures stay blocked. This mask never edits the observation.
        passable = (terrain | goal) & ~home & ~(obs.castles & obs.neutral_cells & ~affordable)
        costs = 1 + jnp.where(~friendly, armies, 0) * 0.12
        to_goal = _distance(passable, goal, costs)
        dest_distance = _neighbors(to_goal, _INF)
        advances = dest_distance < to_goal[..., None]
        moving = jnp.maximum(armies - 1, 0)[..., None]
        dest_army = _neighbors(armies, 0)
        dest_friendly = _neighbors(friendly, False)
        dest_enemy = _neighbors(obs.opponent_cells, False)
        dest_castle = _neighbors(obs.castles, False)
        visible_destination = ~_neighbors(hidden_cells, True)
        captures = ~dest_friendly & (moving > dest_army)
        enemy_force = jnp.where(obs.opponent_cells, jnp.maximum(armies - 1, 0), 0)
        local_threat = jnp.max(_neighbors(enemy_force, 0), axis=-1)
        counterforce = _neighbors(local_threat, 0)
        delivered = jnp.where(dest_friendly, moving + dest_army, moving - dest_army)
        # Full moves only, no own-general sortie. Surviving force must exceed
        # the largest currently visible adjacent counterattack. Merges, growth
        # and hidden threats are not predicted by this one-action check.
        valid = (
            compute_valid_move_mask_obs(obs)
            & ~home[..., None]
            & advances
            & (to_goal[..., None] < _INF / 2)
            & visible_destination
            & ~_neighbors(obs.mountains | obs.structures_in_fog, True)
            & (dest_friendly | captures)
            & (delivered > counterforce)
        )
        scores = 5 + moving * 0.65 + captures * (3 + dest_enemy * 3 + dest_castle * 18)
        scores += dest_friendly * jnp.minimum(dest_army, moving) * 0.10
        scores -= jnp.where(captures, dest_army * 0.12, 0)
        choice = jnp.argmax(jnp.where(valid, scores, -_INF))
        source, direction = choice // 4, choice % 4
        planned = jnp.array([0, source // w, source % w, direction, 0], jnp.int32)
        available = hidden & jnp.any(valid)
        defense = base_memory.defense
        priority = (
            telemetry["offense_defense_priority"]
            | (telemetry["intercept_home_deficit"] > 0)
            | telemetry["commitment_started"]
            | telemetry["commitment_held"]
            | (base_input.defense.defender >= 0)
            | (defense.defender >= 0)
            | (base_action[0] == 2)
            | seen
        )
        safe = _preserves_home(obs, planned, terrain, home, home_distance, self.deathtouch_turn)
        use = available & safe & ~priority
        action = jnp.where(use, planned, base_action)
        changed = use & jnp.any(action != base_action)
        # A V8 offensive packet may only be transported if its action was
        # actually issued. Defense priority prevents overriding its transport.
        cleared = blank_base._replace(defense=defense, last_turn=time)
        base_memory = jax.tree.map(lambda old, blank: jnp.where(changed, blank, old), base_memory, cleared)
        new_memory = StrategicMemory(base_memory, goal_index, time, jnp.int32(h), jnp.int32(w))
        telemetry = telemetry | dict(
            remembered_general_visible=seen,
            remembered_general_available=known,
            remembered_general_hidden=hidden,
            remembered_general_invalidated=invalidated,
            remembered_general_index=goal_index,
            strategic_map_reset=reset,
            strategic_observation_gap=gap & ~reset,
            pursuit_available=available,
            pursuit_issued=use,
            pursuit_override=changed,
            actual_v8_action_issued=~changed,
            pursuit_home_safe=safe,
            pursuit_priority_blocked=available & priority,
            pursuit_base_kind=base_action[0],
            pursuit_base_row=base_action[1],
            pursuit_base_column=base_action[2],
            pursuit_base_direction=base_action[3],
            pursuit_base_split=base_action[4],
        )
        telemetry["building"] = action[0] == 2
        return action, new_memory, telemetry

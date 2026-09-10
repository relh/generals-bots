"""Experimental local fanout from small garrisons instead of owned transfers.

Only visible empty neutral plains are candidates. Spend at most three armies,
keeping the parent general reserve. Tick mode intervenes on the last decision
before whole-land production; fanout mode also expands between ticks. Both stop
after the first remembered hostile contact. Future income depends on actually
retaining the tile. This is not a complete tick optimizer or FFA contact strategy.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs

from .sentinel_agent import _neighbors
from .sentinel_v6_agent import _destination
from .sentinel_v9_agent import StrategicMemory
from .sentinel_v10_agent import SentinelV10Agent

_FIELDS = ("kind", "row", "column", "direction", "split")


class FrontierMemory(NamedTuple):
    base: StrategicMemory
    contact_seen: jax.Array


def _frontier_proposal(obs, reserve=3):
    a, mine = obs.armies, obs.owned_cells
    home = mine & obs.generals
    visible_plain = (obs.neutral_cells & ~obs.castles & ~obs.generals
                     & ~obs.mountains & ~obs.fog_cells & ~obs.structures_in_fog & (a == 0))
    moving = jnp.stack((a - 1, a // 2), axis=-1)[..., None, :]
    remaining = a[..., None, None] - moving
    sources = mine & ~obs.castles
    valid = (compute_valid_move_mask_obs(obs)[..., None] & sources[..., None, None]
             & _neighbors(visible_plain, False)[..., None]
             & (moving >= 1) & (moving <= 3)
             & (~home[..., None, None] | (remaining >= reserve)))
    hr, hc = jnp.argmax(home) // a.shape[1], jnp.argmax(home) % a.shape[1]
    distance = jnp.abs(jnp.arange(a.shape[0])[:, None] - hr) + jnp.abs(jnp.arange(a.shape[1])[None, :] - hc)
    # Lexicographic: smallest expenditure, nearest home, most neutral branches.
    # A large offensive stack cannot be spent merely to acquire one empty tile.
    branches = jnp.sum(_neighbors(visible_plain, False), axis=-1)
    score = (-moving * (5 * a.size + 5) - 5 * _neighbors(distance, a.size)[..., None]
             + _neighbors(branches, 0)[..., None])
    choice = jnp.argmax(jnp.where(valid, score, -100 * a.size))
    source, direction, split = choice // 8, (choice // 2) % 4, choice % 2
    action = jnp.array([0, source // a.shape[1], source % a.shape[1], direction, split], jnp.int32)
    return action, jnp.any(valid) & jnp.any(home), jnp.sum(valid), distance


class SentinelV15Agent:
    """One frozen V10 solve; disabled operation preserves its entire tuple."""

    def __init__(self, id="Sentinel-v15", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, expand_frontier=True, frontier_mode="tick"):
        if frontier_mode not in ("tick", "fanout"):
            raise ValueError("frontier_mode must be tick or fanout")
        self.id = id
        self.expand_frontier = expand_frontier
        self.frontier_mode = frontier_mode
        self.parent = SentinelV10Agent(build_castles=build_castles,
                                      deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        base = self.parent.initial_memory(shape)
        return FrontierMemory(base, jnp.int32(0)) if self.expand_frontier else base

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        if not self.expand_frontier:
            return self.parent.step(obs, key, memory)
        base = memory.base
        time = obs.timestep.astype(jnp.int32)
        reset = ((base.height != obs.armies.shape[0]) | (base.width != obs.armies.shape[1])
                 | (time <= base.last_turn))
        contact = jnp.where(reset, False, memory.contact_seen > 0) | jnp.any(obs.opponent_cells)
        action, returned, tel = self.parent.step(obs, key, base)
        contact |= tel["remembered_general_available"]
        proposed, available, count, local_distance = _frontier_proposal(obs, tel["general_reserve"])
        dest = _destination(action, obs.armies.shape[1], obs.armies.shape)
        transfer = (action[0] == 0) & obs.owned_cells.reshape(-1)[dest]
        offense = ((tel["offense_started"] | tel["offense_continued"])
                   & tel["actual_v8_action_issued"] & tel["mobilization_parent_action_issued"])
        priority = (
            (base.base.defense.defender >= 0) | (returned.base.defense.defender >= 0)
            | tel["intercept_override"] | tel["intercept_guard"] | (tel["intercept_home_deficit"] > 0)
            | tel["commitment_started"] | tel["commitment_continued"] | tel["commitment_held"]
            | tel["commitment_reverse_blocked"] | tel["commitment_override"]
            | (tel["adjacent_threat"] > 0) | (tel["general_reserve"] > tel["general_army"])
            | tel["offense_defense_priority"] | tel["pursuit_issued"] | tel["mobilization_issued"]
            | offense
            | jnp.any(obs.opponent_cells & obs.generals)
        )
        window = jnp.array(self.frontier_mode == "fanout") | (time % 50 == 49)
        eligible = available & window & ~contact & ~priority & (transfer | (action[0] == 1))
        changed = eligible & jnp.any(proposed != action)
        chosen = jnp.where(changed, proposed, action)
        blank = self.parent.initial_memory(obs.armies.shape).base._replace(
            defense=returned.base.defense, last_turn=obs.timestep.astype(jnp.int32),
        )
        returned = returned._replace(base=jax.tree.map(
            lambda old, empty: jnp.where(changed, empty, old), returned.base, blank,
        ))
        proposed_dest = _destination(proposed, obs.armies.shape[1], obs.armies.shape)
        return chosen, FrontierMemory(returned, contact.astype(jnp.int32)), tel | {
            **{"frontier_parent_" + name: action[i] for i, name in enumerate(_FIELDS)},
            **{"frontier_proposal_" + name: proposed[i] for i, name in enumerate(_FIELDS)},
            "frontier_available": available, "frontier_candidate_count": count,
            "frontier_priority": priority, "frontier_eligible": eligible,
            "frontier_contact_seen": contact, "frontier_contact_reset": reset,
            "frontier_window": window, "frontier_issued": changed,
            "frontier_parent_action_issued": ~changed,
            "frontier_next_land_tick": 50 * (obs.timestep // 50 + 1),
            "frontier_steps_to_land_tick": 50 - obs.timestep % 50,
            "frontier_destination_home_manhattan": local_distance.reshape(-1)[proposed_dest],
            "building": chosen[0] == 2,
        }

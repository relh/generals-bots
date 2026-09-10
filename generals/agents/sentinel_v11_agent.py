"""Experimental arbitration between an immediate capture and a new offensive plan.

An ordinary enemy tile is not automatically worth a multi-action detour when
the inherited commander can capture an equally defended enemy tile now. This
is a strategy hypothesis, not a proof that every rejected plan is unproductive.
Existing plans, structures, general pursuit and defensive priorities remain
the parent's decisions. The disabled variant is exactly the frozen V10 parent.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .sentinel_v6_agent import _destination
from .sentinel_v10_agent import SentinelV10Agent

_FIELDS = ("kind", "row", "column", "direction", "split")


class _ReferenceAction:
    """Expose the existing V6 call's proposal without a second policy solve."""

    def __init__(self, agent):
        self.agent = agent

    def initial_memory(self, shape):
        return self.agent.initial_memory(shape)

    def step(self, obs, key, memory):
        action, returned, telemetry = self.agent.step(obs, key, memory)
        return action, returned, telemetry | {
            "capture_reference_" + name: action[i] for i, name in enumerate(_FIELDS)
        }


class SentinelV11Agent:
    def __init__(
        self, id="Sentinel-v11", *, build_castles=False, deathtouch_turn=None,
        max_turns=1200, preserve_enemy_captures=True,
    ):
        self.id = id
        self.preserve_enemy_captures = preserve_enemy_captures
        self.parent = SentinelV10Agent(
            build_castles=build_castles, deathtouch_turn=deathtouch_turn, max_turns=max_turns,
        )
        if preserve_enemy_captures:
            # This candidate owns the entire composition. Instrument its V8
            # commander's V6 call; no frozen class or other instance is changed.
            offensive = self.parent.base.base
            offensive.base = _ReferenceAction(offensive.base)

    def initial_memory(self, shape):
        return self.parent.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.parent.step(obs, key, memory)
        if not self.preserve_enemy_captures:
            return action, returned, telemetry

        reference = jnp.stack([telemetry["capture_reference_" + name] for name in _FIELDS])
        armies = obs.armies
        width = armies.shape[1]
        source = reference[1] * width + reference[2]
        destination = _destination(reference, width, armies.shape)
        moving = jnp.where(reference[4] == 1, armies.reshape(-1)[source] // 2,
                           armies.reshape(-1)[source] - 1)
        enemy_plain = obs.opponent_cells & ~obs.castles & ~obs.generals
        immediate = (
            (reference[0] == 0) & enemy_plain.reshape(-1)[destination]
            & (moving > armies.reshape(-1)[destination])
        )
        objective = jnp.clip(telemetry["offense_objective"], 0, armies.size - 1)
        eta = jnp.where(telemetry["offense_direct_chosen"], telemetry["offense_direct_eta"],
                        telemetry["offense_collection_eta"])
        actual_start = (
            telemetry["offense_started"] & telemetry["actual_v8_action_issued"]
            & telemetry["mobilization_parent_action_issued"]
        )
        prefer = (
            actual_start & (eta > 1) & immediate
            & enemy_plain.reshape(-1)[objective]
            & (armies.reshape(-1)[destination] >= armies.reshape(-1)[objective])
        )
        changed = prefer & jnp.any(reference != action)
        # The unissued offensive plan has no transported packet. Keep the
        # actual V6 defender result and observed stationary-general knowledge.
        cleared = self.initial_memory(armies.shape).base._replace(
            defense=returned.base.defense, last_turn=obs.timestep.astype(jnp.int32),
        )
        returned = returned._replace(base=jax.tree.map(
            lambda old, blank: jnp.where(changed, blank, old), returned.base, cleared,
        ))
        chosen = jnp.where(changed, reference, action)
        telemetry = telemetry | dict(
            capture_priority_issued=changed,
            capture_parent_action_issued=~changed,
            capture_immediate_available=immediate,
            capture_planned_eta=eta,
            capture_planned_defenders=armies.reshape(-1)[objective],
            capture_immediate_defenders=armies.reshape(-1)[destination],
            building=chosen[0] == 2,
        )
        return chosen, returned, telemetry

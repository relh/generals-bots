"""Experimental action-service budget between collection and the V6 campaign.

An actually issued collection transfer charges one turn. An uncommitted V6
campaign move repays one turn. While debt remains, defer a new collection only
if a different eligible campaign move exists. Existing plans and ready direct
deployments retain priority; PASS/build are never forced as repayment.

This is a scheduling hypothesis, not a neutral-expansion or safety guarantee.
Campaign scores include emergency incentives even outside explicit defensive
modes. Same-action collection/campaign agreement charges once and keeps the
plan; there are no action-identical memory cancellations. Inherited events
require service_parent_action_issued before attribution to the final decision.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sentinel_v9_agent import StrategicMemory
from .sentinel_v10_agent import SentinelV10Agent

_FIELDS = ("kind", "row", "column", "direction", "split")


class ServiceMemory(NamedTuple):
    base: StrategicMemory
    collection_debt: jax.Array


def _campaign_move(action, telemetry):
    """An operational class, not a claim that all defensive incentives vanish."""
    explicit_defense = (
        telemetry["intercept_override"] | telemetry["intercept_guard"]
        | telemetry["commitment_started"] | telemetry["commitment_continued"]
        | telemetry["commitment_held"] | telemetry["commitment_reverse_blocked"]
        | telemetry["commitment_override"] | (telemetry["adjacent_threat"] > 0)
        | (telemetry["general_reserve"] > telemetry["general_army"])
    )
    return (action[0] == 0) & ~explicit_defense


class _CampaignReference:
    """Expose the actual inner V6 call and its pre-wrapper attribution."""

    def __init__(self, agent):
        self.agent = agent

    def initial_memory(self, shape):
        return self.agent.initial_memory(shape)

    def step(self, obs, key, memory):
        action, returned, telemetry = self.agent.step(obs, key, memory)
        return action, returned, telemetry | {
            **{"service_reference_" + name: action[i] for i, name in enumerate(_FIELDS)},
            "service_reference_campaign_move": _campaign_move(action, telemetry),
        }


class SentinelV14Agent:
    def __init__(self, id="Sentinel-v14", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, balance_collection=True):
        self.id = id
        self.balance_collection = balance_collection
        self.parent = SentinelV10Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn,
                                      max_turns=max_turns)
        if balance_collection:
            offensive = self.parent.base.base
            offensive.base = _CampaignReference(offensive.base)

    def initial_memory(self, shape):
        base = self.parent.initial_memory(shape)
        return ServiceMemory(base, jnp.int32(0)) if self.balance_collection else base

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        if not self.balance_collection:
            return self.parent.step(obs, key, memory)
        time = obs.timestep.astype(jnp.int32)
        base = memory.base
        reset = ((base.height != obs.armies.shape[0]) | (base.width != obs.armies.shape[1])
                 | (time <= base.last_turn) | ((base.last_turn >= 0) & (time > base.last_turn + 1)))
        debt = jnp.where(reset, 0, memory.collection_debt).astype(jnp.int32)
        action, returned, telemetry = self.parent.step(obs, key, base)
        reference = jnp.stack([telemetry["service_reference_" + name] for name in _FIELDS])
        offense = ((telemetry["offense_started"] | telemetry["offense_continued"])
                   & telemetry["actual_v8_action_issued"] & telemetry["mobilization_parent_action_issued"])
        collecting = offense & telemetry["offense_collecting"] & (action[0] == 0)
        defer = (collecting & telemetry["offense_started"] & (debt > 0)
                 & ~telemetry["pursuit_issued"] & telemetry["service_reference_campaign_move"]
                 & jnp.any(reference != action))
        chosen = jnp.where(defer, reference, action)
        charge = collecting & ~defer
        ordinary = (~offense & ~telemetry["pursuit_issued"]
                    & telemetry["mobilization_parent_action_issued"] & jnp.all(action == reference))
        service = (defer | ordinary) & telemetry["service_reference_campaign_move"] & ~charge
        after = jnp.maximum(debt + charge.astype(jnp.int32) - service.astype(jnp.int32), 0)
        # The deferred collection never transported its proposed packet.
        blank = self.parent.initial_memory(obs.armies.shape).base._replace(
            defense=returned.base.defense, last_turn=time,
        )
        returned = returned._replace(base=jax.tree.map(
            lambda old, empty: jnp.where(defer, empty, old), returned.base, blank,
        ))
        telemetry |= {"service_parent_" + name: action[i] for i, name in enumerate(_FIELDS)}
        telemetry |= dict(
            service_debt_reset=reset, service_debt_before=debt, service_collection_charged=charge,
            service_campaign_served=service, service_collection_deferred=defer,
            service_debt_repaid=jnp.minimum(debt, service.astype(jnp.int32)),
            service_parent_action_issued=~defer, service_debt_after=after, building=chosen[0] == 2,
        )
        return chosen, ServiceMemory(returned, after), telemetry

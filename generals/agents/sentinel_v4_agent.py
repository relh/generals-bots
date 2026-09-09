"""Experimental v2 campaign with a short visible-threat check before building.

The attack envelope only constrains construction sites. It retains no fog
memory and makes no changes to campaign moves or general defense.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .agent import Agent
from .sentinel_agent import SentinelAgent, _build_prices, _distance, _neighbors
from .sentinel_v3_agent import _campaign_decision


def _build_threat(obs, horizon):
    """Visible full-stack envelope within one or two orthogonal moves.

    Known obstacles block intermediate routes. We neither invent hidden armies
    nor credit speculative attrition or friendly reinforcements. The maximum
    describes individual stacks; it does not model enemy stack merging.
    """
    force = jnp.where(obs.opponent_cells, jnp.maximum(obs.armies - 1, 0), 0)
    adjacent = jnp.max(_neighbors(force, 0), axis=-1)
    if horizon == 1:
        return adjacent
    through = jnp.where(~(obs.mountains | obs.structures_in_fog), adjacent, 0)
    return jnp.maximum(adjacent, jnp.max(_neighbors(through, 0), axis=-1))


class SentinelV4Agent(Agent):
    """Stateless candidate; horizon one is the exact frozen-v2 build ablation."""

    def __init__(
        self,
        id="Sentinel-v4",
        *,
        build_castles=False,
        deathtouch_turn=None,
        max_turns=1200,
        build_threat_horizon=2,
    ):
        super().__init__(id)
        if build_threat_horizon not in (1, 2):
            raise ValueError("build_threat_horizon must be 1 or 2")
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.build_threat_horizon = build_threat_horizon
        self.campaign = SentinelAgent(build_castles=False, deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    @partial(jax.jit, static_argnums=0)
    def act(self, obs, key):
        return self.decision(obs, key)[0]

    @partial(jax.jit, static_argnums=0)
    def decision(self, obs, key):
        action, telemetry, _ = _campaign_decision(self.campaign, obs, key)
        telemetry = dict(
            telemetry,
            build_candidate_count=jnp.int32(0),
            build_rejected_threat_count=jnp.int32(0),
            selected_build_threat=jnp.int32(0),
            selected_build_remaining=jnp.int32(0),
        )
        if not self.build_castles:
            return action, telemetry

        a, mine = obs.armies, obs.owned_cells
        structures = mine & (obs.generals | obs.castles)
        biggest = jnp.max(jnp.where(mine, a, 0))
        affordable = obs.castles & obs.neutral_cells & (a + 6 < biggest)
        passable = ~(obs.mountains | obs.structures_in_fog | (obs.castles & obs.neutral_cells & ~affordable))
        home_distance = _distance(passable, mine & obs.generals)
        near_force = jnp.max(jnp.where(obs.opponent_cells & (home_distance <= 3), jnp.maximum(a - home_distance, 0), 0))
        price = _build_prices(structures)
        adjacent = _build_threat(obs, 1)
        threat = adjacent if self.build_threat_horizon == 1 else _build_threat(obs, 2)
        # Preserve all original economic, route and home-defense conditions.
        original_eligible = (
            mine
            & ~structures
            & (a >= price + 5 + adjacent)
            & (home_distance >= 2)
            & (near_force < telemetry["general_army"])
            & (obs.timestep + 2 * price + 100 < self.max_turns)
        )
        eligible = original_eligible & (a >= price + 5 + threat)
        build_score = jnp.where(eligible, 45 + (self.max_turns - obs.timestep) * 0.08 - price * 0.3, -1e9)
        index = jnp.argmax(build_score)
        imminent, home = telemetry["adjacent_threat"], telemetry["general_army"]
        pass_unsafe = (imminent > home) | (telemetry["deathtouch_active"] & (imminent > 0))
        pass_score = jnp.where(pass_unsafe, -2000 - jnp.maximum(imminent - home, 0) * 20, 0)
        building = jnp.max(build_score) > jnp.maximum(telemetry["score"], pass_score)
        width = a.shape[1]
        action = jnp.where(building, jnp.array([2, index // width, index % width, 0, 0], jnp.int32), action)
        telemetry.update(
            building=building,
            build_candidate_count=jnp.sum(eligible),
            build_rejected_threat_count=jnp.sum(original_eligible & ~eligible),
            selected_build_threat=jnp.where(building, threat.reshape(-1)[index], 0),
            selected_build_remaining=jnp.where(building, (a - price).reshape(-1)[index], 0),
        )
        return action, telemetry

"""Experimental postcontact expansion from separate two-army garrisons.

A visible empty neutral plain can receive one army from an ordinary two-army
source while the frozen parent's uncommitted owned transfer yields its action. The main
packet retains its army but loses a move; this is an allocation hypothesis,
not free expansion. Current visible route safety has the inherited radius-ten
scope and does not certify hidden force or future adaptive replies.

Native nineteen-field parent memory and its telemetry are retained. Every
inherited issued-action event also requires ``rear_parent_action_issued``.
"""

from functools import partial

import jax
import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs

from .sentinel_agent import _distance, _neighbors
from .sentinel_v6_agent import _destination
from .sentinel_v7_agent import _preserves_home
from .sentinel_v10_agent import SentinelV10Agent

_FIELDS = ("kind", "row", "column", "direction", "split")
_INF = 1e6


def _rear_proposal(obs, parent_action, home_distance):
    armies = obs.armies
    width = armies.shape[1]
    cells = jnp.arange(armies.size).reshape(armies.shape)
    plain = ~(obs.generals | obs.castles | obs.mountains | obs.fog_cells | obs.structures_in_fog)
    empty = obs.neutral_cells & plain & (armies == 0)
    hostile = jnp.max(_neighbors(jnp.where(obs.opponent_cells, jnp.maximum(armies - 1, 0), 0), 0), axis=-1)
    parent_source = parent_action[1] * width + parent_action[2]
    donors = obs.owned_cells & plain & (armies == 2) & (hostile == 0) & (cells != parent_source)
    targets = empty & (hostile == 0) & (home_distance < _INF / 2)
    valid = compute_valid_move_mask_obs(obs) & donors[..., None] & _neighbors(targets, False)
    branches = jnp.sum(_neighbors(empty, False), axis=-1)
    # Distances are integral; five dominates every possible branch-count tie.
    score = -5 * _neighbors(home_distance, _INF) + _neighbors(branches, 0)
    choice = jnp.argmax(jnp.where(valid, score, -10 * _INF))
    source, direction = choice // 4, choice % 4
    action = jnp.array([0, source // width, source % width, direction, 0], jnp.int32)
    target = _destination(action, width, armies.shape)
    return action, jnp.any(valid), jnp.sum(valid), branches.reshape(-1)[target]


class SentinelV18Agent:
    """One frozen V10 call; disabled operation returns its complete tuple."""

    def __init__(self, id="Sentinel-v18", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, rear_expansion=True):
        self.id = id
        self.rear_expansion = rear_expansion
        self.deathtouch_turn = deathtouch_turn
        self.parent = SentinelV10Agent(build_castles=build_castles,
                                      deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        return self.parent.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        action, returned, telemetry = self.parent.step(obs, key, memory)
        if not self.rear_expansion:
            return action, returned, telemetry

        home = obs.owned_cells & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        home_distance = _distance(terrain, home)
        proposed, available, count, branches = _rear_proposal(obs, action, home_distance)
        target = _destination(proposed, obs.armies.shape[1], obs.armies.shape)
        source = proposed[1] * obs.armies.shape[1] + proposed[2]
        parent_target = _destination(action, obs.armies.shape[1], obs.armies.shape)
        transfer = (action[0] == 0) & obs.owned_cells.reshape(-1)[parent_target]
        contact = jnp.any(obs.opponent_cells)
        visible_general = jnp.any(obs.opponent_cells & obs.generals)
        remembered_general = ((memory.enemy_general >= 0) | (returned.enemy_general >= 0)
                              | telemetry["remembered_general_available"])
        incoming_offense = memory.base.phase > 0
        returned_offense = returned.base.phase > 0
        incoming_defender = memory.base.defense.defender >= 0
        returned_defender = returned.base.defense.defender >= 0
        defense_priority = (
            telemetry["offense_defense_priority"] | telemetry["intercept_override"]
            | telemetry["intercept_guard"] | (telemetry["intercept_home_deficit"] > 0)
            | (telemetry["adjacent_threat"] > 0)
            | (telemetry["general_reserve"] > telemetry["general_army"])
            | telemetry["commitment_started"] | telemetry["commitment_continued"]
            | telemetry["commitment_held"] | telemetry["commitment_reverse_blocked"]
            | telemetry["commitment_override"]
        )
        pursuit = telemetry["pursuit_issued"]
        mobilization = telemetry["mobilization_issued"]
        priority = (incoming_offense | returned_offense | incoming_defender | returned_defender
                    | defense_priority | pursuit | mobilization | visible_general | remembered_general)
        has_home = jnp.any(home)
        eligible = available & has_home & contact & transfer & ~priority
        safe = jax.lax.cond(
            eligible,
            lambda _: _preserves_home(obs, proposed, terrain, home, home_distance, self.deathtouch_turn),
            lambda _: jnp.array(False),
            operand=None,
        )
        issued = eligible & safe
        chosen = jnp.where(issued, proposed, action)
        # Both incoming and returned transport/defender slots are empty whenever
        # arbitration can issue. Preserve the exact actual parent memory tuple.
        return chosen, returned, telemetry | {
            **{"rear_parent_" + name: action[i] for i, name in enumerate(_FIELDS)},
            **{"rear_proposal_" + name: proposed[i] for i, name in enumerate(_FIELDS)},
            "rear_available": available,
            "rear_candidate_count": count,
            "rear_visible_contact": contact,
            "rear_has_home": has_home,
            "rear_visible_general": visible_general,
            "rear_remembered_general": remembered_general,
            "rear_parent_owned_transfer": transfer,
            "rear_incoming_offense": incoming_offense,
            "rear_returned_offense": returned_offense,
            "rear_incoming_defender": incoming_defender,
            "rear_returned_defender": returned_defender,
            "rear_defense_priority": defense_priority,
            "rear_pursuit": pursuit,
            "rear_mobilization": mobilization,
            "rear_priority": priority,
            "rear_eligible": eligible,
            "rear_safety_checked": eligible,
            "rear_home_safe": safe,
            "rear_source": jnp.where(available, source, -1),
            "rear_target": jnp.where(available, target, -1),
            "rear_destination_home_distance": jnp.where(available, home_distance.reshape(-1)[target], -1),
            "rear_destination_neutral_branches": jnp.where(available, branches, -1),
            "rear_issued": issued,
            "rear_override": issued,
            "rear_parent_action_issued": ~issued,
        }

"""Experimental same-packet visible capture chains inside frozen V16.

On a new immediate ordinary-enemy direct attack, compare simple paths of up to
three currently visible plain captures. Maximize captures, then surviving force;
a tie retains the parent's first action. This is a territorial-value hypothesis,
not evidence that a weaker enemy target is strategically better. Collections,
feeders and existing ordinary deployments keep their original admissions.

Phase4 stores at most two fixed pending destinations in rally/objective. Never
pass that phase into V16: its fallback sees blank offense and the real defender.
Every issued step rechecks observed ownership, force, suffix and home coverage.
No enemy movement, growth, fog army, future donation or successful capture is
assumed. Counterforce checks only the largest visible neighboring enemy; merges
and simultaneous movement are not predicted. Home coverage is the inherited
radius-ten approximation checked for the next move, not a full-route guarantee.

Inherited feeder/offense telemetry describes the V16 proposal: mask with
chain_parent_action_issued AND the outer actual_v8_action_issued and
mobilization_parent_action_issued. Chain events also require both outer masks.
An issued target attack is an attempt; next-observation ownership confirms it.
chain_available reports enumeration availability; original comparison scores
are meaningful only when chain_comparison_available, otherwise they are -1.
"""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp

from .sentinel_agent import _distance
from .sentinel_v6_agent import _destination
from .sentinel_v7_agent import _preserves_home
from .sentinel_v16_agent import SentinelV16Agent, _FeederCampaign

_FIELDS = ("kind", "row", "column", "direction", "split")
_ROUTES = tuple((directions + (0,) * (3 - size), size)
                for size in range(1, 4) for directions in product(range(4), repeat=size))


def _route_cells(source, directions, shape):
    """Vectorized coordinates; preserve an explicit off-board validity mask."""
    h, w = shape
    dr = jnp.array([-1, 1, 0, 0], jnp.int32)[directions]
    dc = jnp.array([0, 0, -1, 1], jnp.int32)[directions]
    rows = source // w + jnp.cumsum(dr, axis=-1)
    columns = source % w + jnp.cumsum(dc, axis=-1)
    valid = (rows >= 0) & (rows < h) & (columns >= 0) & (columns < w)
    return jnp.clip(rows, 0, h - 1) * w + jnp.clip(columns, 0, w - 1), valid


def _capture_values(obs, source, paths, lengths, bounds):
    """Evaluate fixed distinct visible capture paths, with no hidden donations."""
    a = obs.armies.reshape(-1)
    w = obs.armies.shape[1]
    source = jnp.clip(source, 0, a.size - 1)
    mine = obs.owned_cells.reshape(-1)
    plain = ~(obs.castles | obs.generals | obs.mountains | obs.structures_in_fog | obs.fog_cells)
    plain &= obs.opponent_cells | obs.neutral_cells
    enabled = jnp.arange(3)[None, :] < lengths[:, None]
    defenders = a[paths]
    residuals = a[source] - jnp.cumsum(jnp.where(enabled, defenders + 1, 0), axis=1)
    # Each destination must be new territory, including within this projection.
    distinct = paths != source
    distinct &= ~jnp.any(
        (paths[:, :, None] == paths[:, None, :])
        & (jnp.arange(3)[None, None, :] < jnp.arange(3)[None, :, None]), axis=-1,
    )
    enemy = obs.opponent_cells.reshape(-1)
    offsets = jnp.array([-w, w, -1, 1], jnp.int32)
    neighbors = paths[..., None] + offsets
    r, c = paths // w, paths % w
    adjacent_ok = jnp.stack((r > 0, r < obs.armies.shape[0] - 1, c > 0, c < w - 1), axis=-1)
    neighbors = jnp.clip(neighbors, 0, a.size - 1)
    # An enemy captured earlier in this chain no longer counterattacks.
    captured = jnp.any(
        (neighbors[..., None] == paths[:, None, None, :])
        & (jnp.arange(3)[None, None, None, :] <= jnp.arange(3)[None, :, None, None]), axis=-1,
    )
    threat = jnp.max(jnp.where(adjacent_ok & enemy[neighbors] & ~captured,
                               jnp.maximum(a[neighbors] - 1, 0), 0), axis=-1)
    force_ok = jnp.all(~enabled | (residuals > threat), axis=-1)
    targets_ok = jnp.all(~enabled | (bounds & distinct & plain.reshape(-1)[paths]), axis=-1)
    source_ok = mine[source] & ~obs.generals.reshape(-1)[source] & (a[source] > 1)
    valid = source_ok & targets_ok & force_ok & (lengths >= 1) & (lengths <= 3)
    final = jnp.take_along_axis(residuals, jnp.clip(lengths - 1, 0, 2)[:, None], axis=1)[:, 0]
    return valid, final, residuals[:, 0], targets_ok, force_ok


def _best_chain(obs, source, parent_direction):
    directions = jnp.array([route[0] for route in _ROUTES], jnp.int32)
    lengths = jnp.array([route[1] for route in _ROUTES], jnp.int32)
    paths, bounds = _route_cells(source, directions, obs.armies.shape)
    valid, residual, first_force, _, _ = _capture_values(obs, source, paths, lengths, bounds)
    # The first target must be an enemy ordinary tile; neutral continuations
    # can supply territory, but cannot steal unrelated neutral expansion.
    valid &= obs.opponent_cells.reshape(-1)[paths[:, 0]]
    original = valid & (directions[:, 0] == parent_direction)
    original_length = jnp.max(jnp.where(original, lengths, 0))
    original_force = jnp.max(jnp.where(original & (lengths == original_length), residual, -1))
    best_length = jnp.max(jnp.where(valid, lengths, 0))
    best_force = jnp.max(jnp.where(valid & (lengths == best_length), residual, -1))
    tied = valid & (lengths == best_length) & (residual == best_force)
    preserve = jnp.any(tied & original)
    chosen = jnp.argmax(tied & (~preserve | original))
    return (paths[chosen], directions[chosen, 0], best_length, best_force, first_force[chosen],
            jnp.any(valid), original_length, original_force)


class _CaptureCampaign(_FeederCampaign):
    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        active = memory.phase == 4
        time = obs.timestep.astype(jnp.int32)
        blank = self.initial_memory(obs.armies.shape)._replace(defense=memory.defense, last_turn=time)
        parent_input = jax.tree.map(lambda old, empty: jnp.where(active, empty, old), memory, blank)
        parent, returned, tel = super().step(obs, key, parent_input)
        a, mine = obs.armies, obs.owned_cells
        w = a.shape[1]
        source = jnp.clip(jnp.where(active, memory.packet, parent[1] * w + parent[2]), 0, a.size - 1)
        paths, direction, length, residual, first_force, available, old_length, old_force = _best_chain(
            obs, source, parent[3],
        )
        parent_dest = _destination(parent, w, a.shape)
        reference = jnp.array([tel[f"feeder_reference_{field}"] for field in _FIELDS], jnp.int32)
        reference_dest = _destination(reference, w, a.shape)
        structures = (obs.castles | obs.generals).reshape(-1)

        def structural_capture(action, dest):
            src = jnp.clip(action[1] * w + action[2], 0, a.size - 1)
            moving = jnp.where(action[4] == 0, a.reshape(-1)[src] - 1, a.reshape(-1)[src] // 2)
            touch = (obs.timestep >= self.deathtouch_turn
                     if self.deathtouch_turn is not None else jnp.array(False))
            winning = obs.opponent_cells.reshape(-1)[dest] & obs.generals.reshape(-1)[dest] & touch & (moving > 0)
            return ((action[0] == 0) & structures[dest] & ~mine.reshape(-1)[dest]
                    & ((moving > a.reshape(-1)[dest]) | winning))

        priority = (
            tel["offense_defense_priority"] | (tel["intercept_home_deficit"] > 0)
            | tel["intercept_guard"] | (tel["adjacent_threat"] > 0)
            | (memory.defense.defender >= 0) | (returned.defense.defender >= 0)
            | (parent[0] == 2) | (reference[0] == 2)
            | structural_capture(parent, parent_dest) | structural_capture(reference, reference_dest)
            | jnp.any(obs.opponent_cells & obs.generals)
        )
        admission = ((memory.phase == 0) & tel["offense_direct_started"] & tel["offense_attack_issued"]
                     & (parent[0] == 0) & (parent[4] == 0)
                     & obs.opponent_cells.reshape(-1)[parent_dest] & ~structures[parent_dest])
        # Incoming phase4 contains fixed targets: rally is next, objective last.
        pending = jnp.array([memory.rally, memory.objective, memory.objective], jnp.int32)
        pending_bounds = (pending >= 0) & (pending < a.size)
        pending = jnp.clip(pending, 0, a.size - 1)
        prior_cells = jnp.array([source, pending[0], pending[1]], jnp.int32)
        adjacent = abs(pending // w - prior_cells // w) + abs(pending % w - prior_cells % w) == 1
        fixed_ok, fixed_residual, fixed_first, targets_ok, force_ok = _capture_values(
            obs, source, pending[None, :], memory.remaining[None], (pending_bounds & adjacent)[None, :],
        )
        gap = memory.last_turn + 1 != time
        expired = time > memory.expires
        consistent = ((memory.packet >= 0) & (memory.packet < a.size)
                      & mine.reshape(-1)[source] & ~obs.generals.reshape(-1)[source]
                      & (a.reshape(-1)[source] > 1) & (a.reshape(-1)[source] >= memory.expected_army)
                      & (memory.expected_army > 0) & (memory.remaining >= 1) & (memory.remaining <= 2))
        paths = jnp.where(active, pending, paths)
        length = jnp.where(active, memory.remaining, length)
        residual = jnp.where(active, fixed_residual[0], residual)
        first_force = jnp.where(active, fixed_first[0], first_force)
        target = paths[0]
        delta_row, delta_col = target // w - source // w, target % w - source % w
        fixed_direction = jnp.where(delta_row < 0, 0, jnp.where(delta_row > 0, 1, jnp.where(delta_col < 0, 2, 3)))
        direction = jnp.where(active, fixed_direction, direction)
        proposal = jnp.array([0, source // w, source % w, direction, 0], jnp.int32)
        home = mine & obs.generals
        terrain = ~(obs.mountains | obs.structures_in_fog)
        # One selected-action home comparison per call; a failed check falls
        # back to V16 instead of searching another first direction.
        home_safe = _preserves_home(obs, proposal, terrain, home, _distance(terrain, home), self.deathtouch_turn)
        continued = active & ~gap & ~expired & consistent & fixed_ok[0]
        started = ~active & admission & available & (length >= 2)
        use = (started | continued) & ~priority & home_safe
        action = jnp.where(use, proposal, parent)
        changed = use & jnp.any(action != parent)
        next_memory = blank._replace(
            defense=returned.defense, packet=target.astype(jnp.int32),
            rally=paths[1].astype(jnp.int32), objective=paths[2].astype(jnp.int32), phase=jnp.int32(4),
            remaining=(length - 1).astype(jnp.int32),
            expires=jnp.where(active, memory.expires, time + length - 1).astype(jnp.int32),
            expected_army=first_force.astype(jnp.int32),
        )
        cleared = blank._replace(defense=returned.defense)
        chain_memory = jax.tree.map(lambda planned, empty: jnp.where(length > 1, planned, empty), next_memory, cleared)
        returned = jax.tree.map(lambda old, new: jnp.where(use, new, old), returned, chain_memory)
        telemetry = tel | dict(
            chain_available=available, chain_admission=admission, chain_priority=priority,
            chain_chosen=started | continued,
            chain_started=use & ~active, chain_continued=use & active,
            chain_issued=use, chain_override=changed, chain_parent_action_issued=~changed,
            chain_released=active & ~use, chain_attack_issued=use, chain_finished=use & (length == 1),
            chain_length=length, chain_residual=residual, chain_next_army=first_force,
            chain_comparison_available=~active & admission,
            chain_original_length=jnp.where(~active & admission, old_length, -1),
            chain_original_residual=jnp.where(~active & admission, old_force, -1),
            chain_packet=source, chain_target=target, chain_next_target=paths[1], chain_last_target=paths[2],
            chain_home_safe=home_safe, chain_reference_defender_active=returned.defense.defender >= 0,
            chain_abort_gap=active & gap, chain_abort_expired=active & expired,
            chain_abort_inconsistent=active & ~consistent,
            chain_abort_route=active & ~targets_ok[0], chain_abort_force=active & ~force_ok[0],
            chain_abort_priority=active & priority, chain_abort_home=active & ~home_safe,
        )
        for label, candidate in (("parent", parent), ("reference", reference), ("proposal", proposal)):
            for index, field in enumerate(_FIELDS):
                telemetry[f"chain_{label}_{field}"] = candidate[index]
        telemetry["building"] = action[0] == 2
        return action, returned, telemetry


class SentinelV17Agent:
    """Disabled returns the exact frozen V16 tuple with native19 memory."""

    def __init__(self, id="Sentinel-v17", *, build_castles=False, deathtouch_turn=None,
                 max_turns=1200, capture_chains=True):
        self.id = id
        self.build_castles = build_castles
        self.deathtouch_turn = deathtouch_turn
        self.max_turns = max_turns
        self.capture_chains = capture_chains
        self.base = SentinelV16Agent(build_castles=build_castles, deathtouch_turn=deathtouch_turn,
                                    max_turns=max_turns)
        if capture_chains:
            self.base.base.base.base = _CaptureCampaign(build_castles=build_castles,
                                                       deathtouch_turn=deathtouch_turn, max_turns=max_turns)

    def initial_memory(self, shape):
        return self.base.initial_memory(shape)

    @partial(jax.jit, static_argnums=0)
    def step(self, obs, key, memory):
        return self.base.step(obs, key, memory)

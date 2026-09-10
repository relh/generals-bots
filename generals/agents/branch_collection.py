"""Experimental, observation-only collection on one deterministic owned tree.

The caller supplies the eligible owned mask, including the rally and excluding
home or unsafe donors. There is no combat, growth, deadline or policy arbitration
here. Results describe exact action budgets, not a complete defense certificate.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .sentinel_agent import _neighbors


class BranchCollectionPlan(NamedTuple):
    delivered: jax.Array
    largest_garrison: jax.Array
    first_source: jax.Array
    first_direction: jax.Array
    feasible: jax.Array
    parent: jax.Array
    distance: jax.Array


def plan_branch_collection(armies, eligible, rally_index):
    """Return rally results for exact budgets 0..6 and the selected tree.

    Inputs are equally shaped 2-D arrays and a flat rally index. Shortest-owned
    parents break ties UP/DOWN/LEFT/RIGHT. Tree children execute in that same
    cardinal order, in postorder. Knapsack ties retain the existing state, then
    the first child budget examined (ascending). Invalid budgets use -1 fields.
    ``delivered`` includes the rally's original army. Every selected edge costs
    one action and leaves one army behind. A child can connect only after its
    selected subtree has gathered more than one army.
    """
    armies = jnp.asarray(armies, jnp.int32)
    eligible = jnp.asarray(eligible, bool)
    cells = jnp.arange(armies.size, dtype=jnp.int32).reshape(armies.shape)
    root = (cells == rally_index) & eligible
    distance = jnp.where(root, 0, 99)

    def spread(_, field):
        candidate = jnp.min(_neighbors(field, 99), axis=-1) + 1
        return jnp.where(eligible, jnp.minimum(field, candidate), 99)

    distance = jax.lax.fori_loop(0, 6, spread, distance)
    reachable = eligible & (distance <= 6)
    parent_direction = jnp.argmax(_neighbors(distance, 99) == distance[..., None] - 1, axis=-1)
    neighbor_cells = _neighbors(cells, -1)
    parent = jnp.take_along_axis(neighbor_cells, parent_direction[..., None], axis=-1)[..., 0]
    parent = jnp.where(reachable & ~root, parent, -1)
    n = armies.size
    budgets = jnp.arange(7)
    empty = jnp.full((n, 7), -1, jnp.int32)
    # All nodes at a depth are independent; children have already been solved.
    def solve_depth(iteration, tables):
        depth = 6 - iteration
        delivered, largest, first, direction = tables
        base = empty.at[:, 0].set(armies.reshape(-1))
        local = (base, base, empty, empty)

        def merge_child(cardinal, current):
            child = neighbor_cells.reshape(n, 4)[:, cardinal]
            safe_child = jnp.maximum(child, 0)
            linked = (child >= 0) & (parent.reshape(-1)[safe_child] == cells.reshape(-1))
            child_delivery = delivered[safe_child]
            child_largest = largest[safe_child]
            child_first = first[safe_child]
            child_direction = direction[safe_child]
            # Each candidate draws from the pre-merge table, never from another
            # candidate that already used this child.
            old_delivery, old_largest, old_first, old_direction = current

            def allocation(child_budget, best):
                previous = budgets - child_budget - 1
                previous_safe = jnp.maximum(previous, 0)
                before = old_delivery[:, previous_safe]
                addition = child_delivery[:, child_budget]
                valid = linked[:, None] & (previous[None, :] >= 0) & (before >= 0) & (addition[:, None] > 1)
                value = before + addition[:, None] - 1
                improve = valid & (value > best[0])
                size = jnp.maximum(old_largest[:, previous_safe], child_largest[:, child_budget, None])
                child_start = jnp.where(child_budget == 0, child, child_first[:, child_budget])
                child_move = jnp.where(
                    child_budget == 0, parent_direction.reshape(-1)[safe_child], child_direction[:, child_budget])
                start = jnp.where(previous[None, :] > 0, old_first[:, previous_safe], child_start[:, None])
                move = jnp.where(previous[None, :] > 0, old_direction[:, previous_safe], child_move[:, None])
                return tuple(jnp.where(improve, candidate, incumbent)
                             for candidate, incumbent in zip((value, size, start, move), best))

            return jax.lax.fori_loop(0, 6, allocation, current)

        local = jax.lax.fori_loop(0, 4, merge_child, local)
        selected = (reachable & (distance == depth)).reshape(-1, 1)
        return tuple(jnp.where(selected, new, old) for new, old in zip(local, tables))

    delivered, largest, first, direction = jax.lax.fori_loop(0, 7, solve_depth, (empty,) * 4)
    safe_root = jnp.clip(rally_index, 0, n - 1)
    valid_root = jnp.any(root)
    def extract(table):
        return jnp.where(valid_root, table[safe_root], -1)

    result = extract(delivered)
    return BranchCollectionPlan(
        result, extract(largest), extract(first), extract(direction), result >= 0, parent, distance)

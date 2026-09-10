"""Independent exhaustive subtree oracle for the experimental gather planner."""
from collections import deque
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.branch_collection import plan_branch_collection


def oracle(armies, eligible, rally):
    h, w = armies.shape
    def neighbors(cell):
        r, c = divmod(cell, w)
        return [(d, (r + dr) * w + c + dc) for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1)))
                if 0 <= r + dr < h and 0 <= c + dc < w]
    distance = {}
    if 0 <= rally < armies.size and eligible.flat[rally]:
        distance[rally] = 0
        queue = deque([rally])
        while queue:
            cell = queue.popleft()
            if distance[cell] == 6:
                continue
            for _, other in neighbors(cell):
                if eligible.flat[other] and other not in distance:
                    distance[other] = distance[cell] + 1
                    queue.append(other)
    parents = {cell: next((d, other) for d, other in neighbors(cell) if distance.get(other) == depth - 1)
               for cell, depth in distance.items() if cell != rally}
    answers = [[] for _ in range(7)]
    if not distance:
        return parents, answers
    for budget in range(7):
        for subset in combinations(parents, budget):
            selected = set(subset) | {rally}
            if any(parents[cell][1] not in selected for cell in subset):
                continue
            actions = []
            def collect(cell):
                total = int(armies.flat[cell])
                for _, child in neighbors(cell):
                    if child in selected and child != rally and parents[child][1] == cell:
                        amount = collect(child)
                        if amount <= 1:
                            raise ValueError('non-executable transfer')
                        actions.append((child, parents[child][0]))
                        total += amount - 1
                return total
            try:
                total = collect(rally)
            except ValueError:
                continue
            answers[budget].append((total, max(int(armies.flat[cell]) for cell in selected),
                                    actions[0] if actions else (-1, -1)))
    return parents, answers


def check(armies, eligible, rally):
    armies, eligible = np.array(armies, np.int32), np.array(eligible, bool)
    plan = plan_branch_collection(armies, eligible, rally)
    parents, answers = oracle(armies, eligible, rally)
    expected_parent = np.full(armies.size, -1)
    for cell, (_, parent) in parents.items():
        expected_parent[cell] = parent
    np.testing.assert_array_equal(plan.parent.reshape(-1), expected_parent)
    for budget, candidates in enumerate(answers):
        assert bool(plan.feasible[budget]) == bool(candidates)
        if not candidates:
            assert all(int(field[budget]) == -1 for field in plan[:4])
        else:
            best = max(row[0] for row in candidates)
            actual = (int(plan.delivered[budget]), int(plan.largest_garrison[budget]),
                      (int(plan.first_source[budget]), int(plan.first_direction[budget])))
            assert actual[0] == best
            assert actual in candidates
    return plan


def test_branch_only_sufficient_fork_and_leavebehind():
    plan = check([[4, 1, 4]], [[1, 1, 1]], 1)
    assert list(map(int, plan.delivered[:3])) == [1, 4, 7]
    assert int(plan.largest_garrison[2]) == 4
    assert (int(plan.first_source[2]), int(plan.first_direction[2])) == (0, 3)
    assert int(plan.delivered[2]) > 6 > int(plan.delivered[1])


def test_army_one_transit_requires_child_first():
    plan = check([[5, 1, 1]], [[1, 1, 1]], 2)
    assert not bool(plan.feasible[1])
    assert int(plan.delivered[2]) == 5
    assert int(plan.first_source[2]) == 0


def test_zero_garrison_transit_receives_then_transfers():
    plan = check([[4, 0, 1]], [[1, 1, 1]], 2)
    assert not bool(plan.feasible[1])
    assert int(plan.delivered[2]) == 3
    assert (int(plan.first_source[2]), int(plan.first_direction[2])) == (0, 3)


def test_zero_garrison_transit_stalls_with_insufficient_arrival():
    plan = check([[2, 0, 1]], [[1, 1, 1]], 2)
    assert not bool(plan.feasible[1])
    assert not bool(plan.feasible[2])


@pytest.mark.parametrize('rally', [0, 4, 8, -1, 9])
def test_sparse_blocked_and_invalid_rally(rally):
    check([[8, 1, 4], [2, 9, 1], [3, 1, 6]], [[1, 0, 1], [1, 1, 0], [1, 1, 1]], rally)


def test_exhaustive_tiny_random_trees_no_overlapping_credit():
    rng = np.random.default_rng(824)
    for _ in range(24):
        check(rng.integers(0, 9, (3, 3)), rng.random((3, 3)) > .2, int(rng.integers(9)))


def test_budget_depth_boundary():
    plan = check([[2] * 8], [[1] * 8], 0)
    assert int(plan.delivered[6]) == 8
    assert int(plan.parent[0, 7]) == -1


def test_jit_vmap_and_determinism():
    armies = jnp.array([[4, 1, 4], [1, 2, 1], [3, 1, 3]], jnp.int32)
    eligible = jnp.ones_like(armies, dtype=bool)
    plain = plan_branch_collection(armies, eligible, 4)
    compiled = jax.jit(plan_branch_collection)(armies, eligible, jnp.int32(4))
    batched = jax.jit(jax.vmap(plan_branch_collection))(jnp.stack([armies] * 2), jnp.stack([eligible] * 2), jnp.array([4, 4]))
    for a, b, c in zip(plain, compiled, batched):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(c, np.stack([a, a]))

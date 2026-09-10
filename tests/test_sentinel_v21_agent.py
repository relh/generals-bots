"""Tree ownership inside frozen V9/V10; synthetic engines are not strength evidence."""

import copy
import hashlib
import io
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from competition.agents.sentinel_python.main import read_observation
from generals.agents.sentinel_v7_agent import OffensiveMemory
from generals.agents.sentinel_v10_agent import SentinelV10Agent
from generals.agents.sentinel_v21_agent import SentinelV21Agent
from generals.core import game
from tests.test_multiplayer import give
from tests.test_sentinel_agent import board
from tests.test_sentinel_v8_agent import same_tree
from tests.test_sentinel_v9_agent import hidden, known_memory
from tests.test_sentinel_v10_agent import restore
from tests.test_sentinel_v20_agent import fork, owned_move

KEY = jax.random.PRNGKey(17)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)
AGENT = SentinelV21Agent(deathtouch_turn=800)
RULES = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
ARCHIVE_AGENT = SentinelV21Agent(**RULES)
DISABLED = SentinelV21Agent(**RULES, tree_collection=False)
PARENT = SentinelV10Agent(**RULES)


def ordinary_memory(memory):
    """Remove only the declared new nested tree, preserving every ordinary field."""
    return memory._replace(base=OffensiveMemory(*(getattr(memory.base, k) for k in OffensiveMemory._fields)))


def issued_masks(tel):
    assert bool(tel["tree_action_issued"]) == bool(
        tel["tree_collector_selected"] & tel["actual_v8_action_issued"] & tel["mobilization_parent_action_issued"]
    )


def embed_parent(agent, shape, parent, tree=None):
    template = agent.initial_memory(shape)
    data = copy.deepcopy(parent)
    data["base"]["tree"] = (
        {k: int(getattr(template.base.tree, k)) for k in template.base.tree._fields} if tree is None else tree
    )
    return restore(template, data)


@pytest.mark.parametrize(
    "name", [f"v10-amin-{t}" for t in (348, 352, 359, 362)] + [f"v10-juraj-{t}" for t in (348, 420)]
)
def test_disabled_preserves_exact_original_v10_three_component_tuple(name):
    folder = Path(__file__).parent / "fixtures/branch_collection"
    path = folder / (name + ".json")
    manifest = json.loads((folder / "manifest.json").read_text())["fixtures"][path.name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
    data = json.loads(path.read_text())
    assert hashlib.sha256(data["public_wire"].encode()).hexdigest() == data["wire_sha256"]
    obs = read_observation(io.StringIO(data["public_wire"]), *data["shape"])
    before = restore(DISABLED.initial_memory(data["shape"]), data["incoming_memory"])
    action, returned, tel = DISABLED.step(obs, jnp.asarray(data["action_key"], jnp.uint32), before)
    np.testing.assert_array_equal(action, data["expected_parent_action"])
    same_tree(returned, restore(returned, data["expected_parent_memory"]))
    assert tel.keys() == data["expected_parent_telemetry"].keys()
    for key, value in data["expected_parent_telemetry"].items():
        np.testing.assert_array_equal(tel[key], value, err_msg=key)
    assert len(jax.tree.leaves(returned)) == 19


def six_edge_state():
    """Declared engine board; first enemy move creates unrelated ordinary collection."""
    grid = jnp.zeros((9, 16), jnp.int32).at[0, 0].set(1).at[8, 15].set(2)
    # Connected terrain detour puts the visible target beyond the ten-step guard horizon.
    grid = grid.at[1, :7].set(-2)
    state = game.create_initial_state(grid)._replace(time=jnp.int32(101))
    for row, col, army in [
        (0, 0, 100),
        (4, 4, 5),
        (4, 5, 1),
        (3, 4, 2),
        (2, 4, 10),
        (5, 4, 2),
        (6, 4, 10),
        (4, 3, 2),
        (4, 2, 10),
        (2, 7, 3),  # Isolated pool witness; no owned path to rally or objective.
        (4, 13, 20),
        (5, 13, 25),
        (4, 14, 1),
    ]:
        state = give(state, 0, (row, col), army)
    state = give(state, 1, (4, 6), 24)
    return give(state, 1, (3, 15), 41)


def test_full_stack_six_real_gathers_survive_unrelated_collection_and_capture():
    state = six_edge_state()
    initial_obs = game.get_observation(state, 0)
    assert public_target_pool(initial_obs, (4, 6)) == 27
    assert terrain_distance(initial_obs.mountains | initial_obs.structures_in_fog, (0, 0), (4, 6)) == 12
    memory = AGENT.initial_memory(state.armies.shape)
    selected_sources, expiry, competing_serial = set(), None, False
    gathers = deployments = attacks = 0
    for step in range(8):
        obs = game.get_observation(state, 0)
        action, returned, tel = AGENT.step(obs, KEY, memory)
        issued_masks(tel)
        assert tel["tree_action_issued"], (step, action, tel)
        assert returned.base.phase == 0  # Unissued ordinary transport owns no memory.
        assert int(tel["tree_objective"]) == 4 * 16 + 6
        assert not tel["tree_parent_action_issued"]
        if step == 0:
            assert tel["tree_started"] and tel["tree_selected_budget"] == 6
            assert tel["tree_delivered"] == 35 and tel["tree_required"] == 27
            assert not tel["tree_single_path_sufficient"]
            expiry = int(returned.base.tree.expires)
        else:
            assert tel["tree_continued"]
        if tel["offense_started"] and tel["offense_collecting"]:
            competing_serial = True
            assert tel["offense_objective"] != tel["tree_objective"]
        _, row, col, direction, split = map(int, action)
        assert split == 0 and state.ownership[0, row, col] and state.armies[row, col] > 1
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[direction]
        target = (row + dr, col + dc)
        if step < 6:
            assert tel["tree_collecting"] and not tel["tree_deploying"]
            assert state.ownership[0, target[0], target[1]]
            selected_sources.add((row, col))
            assert returned.base.tree.budget == 5 - step
            gathers += 1
        else:
            assert tel["tree_deploying"] and not tel["tree_collecting"]
            deployments += 1
        if not tel["tree_attack_issued"]:
            assert returned.base.tree.expires == expiry
        attacks += int(tel["tree_attack_issued"])
        opponent = jnp.array([0, 3, 15, 1, 0], jnp.int32) if step == 0 else PASS
        state, _ = game.step(state, jnp.stack((action, opponent)))
        if not tel["tree_attack_issued"]:
            packet = int(returned.base.tree.packet)
            assert state.ownership[0].reshape(-1)[packet]
            assert state.armies.reshape(-1)[packet] >= returned.base.tree.expected_army
        if step == 5:
            assert state.armies[4, 4] == 35
        memory = returned
    assert (gathers, deployments, attacks) == (6, 2, 1)
    assert competing_serial and len(selected_sources) == 6
    assert state.ownership[0, 4, 6] and state.armies[4, 6] == 10
    assert memory.base.tree.phase == memory.base.phase == 0
    assert memory.base.tree.packet == memory.base.tree.objective == -1
    assert all(x.shape == () and x.dtype == jnp.int32 for x in jax.tree.leaves(memory))
    assert len(jax.tree.leaves(memory)) == 28


def test_ready_direct_handoff_preserves_real_ordinary_memory_and_engine_capture():
    grid = jnp.zeros((7, 7), jnp.int32).at[0, 0].set(1).at[6, 6].set(2)
    state = game.create_initial_state(grid)._replace(time=jnp.int32(101))
    for row, col, army in [(0, 0, 100), (4, 4, 5), (3, 4, 4), (5, 4, 4), (4, 5, 1)]:
        state = give(state, 0, (row, col), army)
    state = give(state, 1, (4, 6), 7)
    memory = AGENT.initial_memory((7, 7))
    for step in range(4):
        obs = game.get_observation(state, 0)
        parent = None
        if step == 2:
            parent_agent = SentinelV10Agent(deathtouch_turn=800)
            parent = parent_agent.step(obs, KEY, ordinary_memory(memory))
        action, returned, tel = AGENT.step(obs, KEY, memory)
        issued_masks(tel)
        if step < 2:
            assert tel["tree_action_issued"] and tel["tree_collecting"]
        if step == 2:
            assert tel["tree_ready_handoff"] and tel["offense_direct_started"]
            assert not tel["tree_action_issued"] and returned.base.tree.phase == 0
            same_tree((action, ordinary_memory(returned)), parent[:2])
        state, _ = game.step(state, jnp.stack((action, PASS)))
        memory = returned
    assert state.ownership[0, 4, 6] and state.armies[4, 6] == 3


def original_context(stem, turn):
    name = "v20_juraj291" if stem == "v20-juraj-v35-game-0012" else "v20_mybot222"
    path = Path(__file__).parent / "fixtures/tree_collection" / (name + ".json")
    expected_hash = {
        "v20_juraj291": "566381002d20d25614a6adaff94e5db0f206009d33087cfe5a9e6c9d1402be1c",
        "v20_mybot222": "fccf6e4435fc14f65008e9d8e5aaa4dc132a16876d9d56b0414f45498b83ed75",
    }
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash[name]
    data = json.loads(path.read_text())
    assert data["turn"] == turn
    assert hashlib.sha256(data["public_wire"].encode()).hexdigest() == data["wire_sha256"]
    parent = data["memory_before"]["parent"]
    shape = tuple(data["shape"])
    assert shape == (parent["height"], parent["width"])
    obs = read_observation(io.StringIO(data["public_wire"]), *shape)
    tree = {k: v for k, v in data["memory_before"].items() if k != "parent"}
    # This explicit schema projection is a new test input, never an archived V21 call.
    memory = embed_parent(ARCHIVE_AGENT, shape, parent, tree)
    return data, obs, jnp.asarray(data["action_key"], jnp.uint32), memory


def test_actual_same_objective_recipient_serial_handoff_keeps_productive_plan():
    call, obs, key, memory = original_context("v20-juraj-v35-game-0012", 291)
    assert memory.base.tree.packet == 279 and memory.base.tree.objective == 219
    action, returned, tel = ARCHIVE_AGENT.step(obs, key, memory)
    issued_masks(tel)
    assert tel["tree_serial_handoff"] and not tel["tree_action_issued"]
    assert tel["offense_collection_eta"] == 6 and tel["offense_remaining"] == 4
    assert returned.base.tree.phase == 0
    np.testing.assert_array_equal(action, [0, 14, 13, 0, 0])
    archived = restore(PARENT.initial_memory(obs.armies.shape), call["memory_after"]["parent"])
    same_tree(ordinary_memory(returned), archived)


@pytest.mark.parametrize("budget,expiry", [(3, 300), (5, 295)])
def test_serial_cannot_claim_exhausted_tree_budget_or_missed_deadline(budget, expiry):
    _, obs, key, memory = original_context("v20-juraj-v35-game-0012", 291)
    memory = memory._replace(
        base=memory.base._replace(tree=memory.base.tree._replace(budget=jnp.int32(budget), expires=jnp.int32(expiry)))
    )
    _, returned, tel = ARCHIVE_AGENT.step(obs, key, memory)
    assert not tel["tree_serial_handoff"]
    if not tel["tree_action_issued"]:
        assert returned.base.tree.phase == 0  # Fresh ordinary fallback is not tree completion.
    issued_masks(tel)


def test_productive_serial_handoff_cannot_extend_original_deadline():
    # Declared current state, not a claimed historical tree: source5→middle3→rally5
    # delivers11 in two gathers, then two deployment moves against enemy7.
    # Both feasible tree and fresh ordinary serial therefore need exactly four actions.
    obs = board(
        [
            (0, 0, 100, True),
            (4, 2, 5, False),
            (4, 3, 3, False),
            (4, 4, 5, False),
            (4, 5, 1, False),
            (2, 6, 5, False),
        ],  # Isolated surplus4 makes target pool10; no collection-route credit.
        [(4, 6, 7, False)],
        shape=(7, 7),
        time=101,
    )
    assert public_target_pool(obs, (4, 6)) == 10
    memory = AGENT.initial_memory(obs.armies.shape)
    tree = memory.base.tree._replace(
        objective=jnp.int32(34),
        rally=jnp.int32(32),
        packet=jnp.int32(30),
        phase=jnp.int32(1),
        budget=jnp.int32(2),
        expires=jnp.int32(104),
        last_turn=jnp.int32(100),
        expected_army=jnp.int32(5),
        remaining=jnp.int32(2),
    )
    memory = memory._replace(last_turn=jnp.int32(100), base=memory.base._replace(last_turn=jnp.int32(100), tree=tree))
    parent = SentinelV10Agent(deathtouch_turn=800).step(obs, KEY, ordinary_memory(memory))
    action, returned, tel = AGENT.step(obs, KEY, memory)
    assert tel["tree_serial_handoff"] and not tel["tree_action_issued"]
    assert tel["tree_remaining_actions"] == tel["offense_collection_eta"] == 4
    assert tel["offense_remaining"] == 2 and parent[1].base.expires == 106
    assert returned.base.expires == 104 and returned.base.tree.phase == 0
    np.testing.assert_array_equal(action, [0, 4, 2, 3, 0])
    expected = parent[1]._replace(base=parent[1].base._replace(expires=jnp.int32(104)))
    same_tree(ordinary_memory(returned), expected)


def test_actual_ready_direct_context_survives_active_unrelated_tree():
    call, obs, key, memory = original_context("v20-my-bot9-game-0004", 222)
    action, returned, tel = ARCHIVE_AGENT.step(obs, key, memory)
    assert tel["tree_ready_handoff"] and tel["offense_direct_started"]
    assert not tel["tree_action_issued"] and returned.base.tree.phase == 0
    np.testing.assert_array_equal(action, call["raw_action"])
    same_tree(
        ordinary_memory(returned), restore(PARENT.initial_memory(obs.armies.shape), call["memory_after"]["parent"])
    )


def started_fork():
    obs = fork()
    action, memory, tel = AGENT.step(obs, KEY, AGENT.initial_memory(obs.armies.shape))
    assert tel["tree_action_issued"]
    return owned_move(obs, action), memory


@pytest.mark.parametrize("condition", ["expired", "missing_recipient", "observation_gap"])
def test_invalid_tree_does_not_continue(condition):
    obs, memory = started_fork()
    tree = memory.base.tree
    if condition == "expired":
        memory = memory._replace(base=memory.base._replace(tree=tree._replace(expires=obs.timestep - 1)))
    elif condition == "missing_recipient":
        row, col = divmod(int(tree.packet), obs.armies.shape[1])
        obs = obs._replace(armies=obs.armies.at[row, col].set(1))
    else:
        obs = obs._replace(timestep=obs.timestep + 1)
    _, returned, tel = AGENT.step(obs, KEY, memory)
    assert not tel["tree_continued"]
    if condition != "observation_gap":
        assert returned.base.tree.phase == 0
    else:
        assert tel["strategic_observation_gap"]
        assert not tel["tree_action_issued"] or tel["tree_started"]


def test_map_reset_clears_nested_tree_and_stale_defense_before_new_admission():
    _, stale = started_fork()
    stale = stale._replace(base=stale.base._replace(defense=stale.base.defense._replace(defender=jnp.int32(32))))
    obs = fork(time=0)
    fresh = AGENT.step(obs, KEY, AGENT.initial_memory(obs.armies.shape))
    reset = AGENT.step(obs, KEY, stale)
    same_tree(reset[:2], fresh[:2])
    assert reset[2]["strategic_map_reset"] and not reset[2]["tree_continued"]


def threatened(obs):
    armies = obs.armies.at[0, 1].set(150)
    enemy = obs.opponent_cells.at[0, 1].set(True)
    return obs._replace(
        armies=armies,
        opponent_cells=enemy,
        neutral_cells=obs.neutral_cells.at[0, 1].set(False),
        opponent_land_count=enemy.sum(),
        opponent_army_count=jnp.sum(jnp.where(enemy, armies, 0)),
    )


def test_batched_active_tree_and_urgent_defense_have_independent_ownership():
    obs, memory = started_fork()
    urgent = threatened(obs)
    batch = jax.tree.map(lambda a, b: None if a is None else jnp.stack((a, b)), obs, urgent)
    memories = jax.tree.map(lambda x: jnp.stack((x, x)), memory)
    _, returned, tel = jax.jit(jax.vmap(AGENT.step))(batch, jnp.stack((KEY, KEY)), memories)
    np.testing.assert_array_equal(tel["tree_action_issued"], [True, False])
    assert tel["tree_urgent_priority"][1] and returned.base.tree.phase[1] == 0
    assert all(x.dtype == jnp.int32 and x.shape == (2,) for x in jax.tree.leaves(returned))
    assert len(jax.tree.leaves(returned)) == 28


def test_outer_mobilization_clears_nested_tree_using_real_saved_public_input():
    data = json.loads((Path(__file__).parent / "fixtures/mobilization-public-1027.json").read_text())
    obs = read_observation(io.StringIO(data["wire"]), 18, 21)
    memory = embed_parent(ARCHIVE_AGENT, (18, 21), data["memory_before"])
    tree = memory.base.tree._replace(phase=jnp.int32(1), packet=jnp.int32(0), last_turn=obs.timestep - 1)
    memory = memory._replace(base=memory.base._replace(tree=tree))
    action, returned, tel = ARCHIVE_AGENT.step(obs, jnp.asarray(data["action_key"], jnp.uint32), memory)
    np.testing.assert_array_equal(action, [0, 14, 5, 0, 0])
    assert tel["mobilization_issued"] and not tel["mobilization_parent_action_issued"]
    assert not tel["tree_action_issued"] and returned.base.tree.phase == returned.base.phase == 0
    issued_masks(tel)


def terrain_distance(mountains, source, target):
    """Independent cardinal BFS on declared public terrain."""
    blocked = np.asarray(mountains)
    frontier, seen = [(source, 0)], {source}
    for (row, col), distance in frontier:
        if (row, col) == target:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + dr, col + dc)
            if (
                0 <= neighbor[0] < blocked.shape[0]
                and 0 <= neighbor[1] < blocked.shape[1]
                and not blocked[neighbor]
                and neighbor not in seen
            ):
                seen.add(neighbor)
                frontier.append((neighbor, distance + 1))
    return None


def public_target_pool(obs, target):
    """Independent scalar check of the original seven-by-seven eligibility pool."""
    armies, mine, enemy = map(np.asarray, (obs.armies, obs.owned_cells, obs.opponent_cells))
    total = 0
    for row, col in np.argwhere(mine & ~np.asarray(obs.generals)):
        if abs(int(row) - target[0]) > 3 or abs(int(col) - target[1]) > 3:
            continue
        hostile = any(
            0 <= row + dr < armies.shape[0]
            and 0 <= col + dc < armies.shape[1]
            and enemy[row + dr, col + dc]
            and armies[row + dr, col + dc] > 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if not hostile:
            total += max(int(armies[row, col]) - 1, 0)
    return total


def pursuit_tree_board(*, different_action):
    owned = [
        (0, 0, 100, True),
        (4, 4, 5, False),
        (4, 5, 1, False),
        (3, 4, 2, False),
        (2, 4, 10, False),
        (5, 4, 2, False),
        (6, 4, 10, False),
        (4, 3, 2, False),
        (4, 2, 10, False),
        (2, 7, 3, False),  # Isolated eligible surplus; no owned collection edge.
    ]
    open_cells = {(r, c) for r, c, _, _ in owned} | {(4, 6), (4, 7), (4, 8)}
    if different_action:
        # Bottom-right neutral route has the same six-edge geometric goal distance
        # as bottom-up owned route, but capture scores +3 instead of owned +0.2.
        open_cells |= {(6, 5), (6, 6), (5, 6)}
    walls = [(r, c) for r in range(9) for c in range(10) if (r, c) not in open_cells]
    obs = board(owned, [(4, 6, 24, False)], mountains=walls, shape=(9, 10), time=101)
    obs = hidden(obs, target=(4, 8))
    assert public_target_pool(obs, (4, 6)) == 27
    return obs


def test_outer_remembered_pursuit_clears_nested_tree_and_masks_collection():
    obs = pursuit_tree_board(different_action=True)
    memory = known_memory(AGENT, obs, target=(4, 8))
    action, returned, tel = AGENT.step(obs, KEY, memory)
    np.testing.assert_array_equal(action, [0, 6, 4, 3, 0])
    ordinary = [int(tel["pursuit_base_" + field]) for field in ("kind", "row", "column", "direction", "split")]
    assert ordinary == [0, 2, 4, 1, 0]
    assert tel["tree_collector_selected"] and tel["tree_started"] and tel["tree_collecting"]
    assert tel["pursuit_issued"] and tel["pursuit_override"] and not tel["actual_v8_action_issued"]
    assert not tel["tree_action_issued"] and returned.base.tree.phase == returned.base.phase == 0
    assert returned.base.tree.packet == returned.base.tree.objective == -1
    issued_masks(tel)


def test_same_physical_pursuit_preserves_actual_tree_ownership():
    obs = pursuit_tree_board(different_action=False)
    memory = known_memory(AGENT, obs, target=(4, 8))
    action, returned, tel = AGENT.step(obs, KEY, memory)
    # Independently declared route: top leaf has only its owned downward exit.
    np.testing.assert_array_equal(action, [0, 2, 4, 1, 0])
    assert tel["pursuit_issued"] and not tel["pursuit_override"]
    assert tel["tree_collector_selected"] and tel["tree_action_issued"]
    assert tel["actual_v8_action_issued"] and tel["mobilization_parent_action_issued"]
    assert returned.base.tree.phase == 1 and returned.base.phase == 0
    assert returned.base.tree.packet == 34 and returned.base.tree.expected_army == 11
    issued_masks(tel)


@pytest.mark.parametrize("winning_general", [False, True])
def test_actual_build_or_winning_capture_preempts_active_tree(winning_general):
    if winning_general:
        obs = board([(0, 0, 20, True), (2, 2, 20, False)], [(2, 3, 10, True)], time=10)
    else:
        obs = board([(0, 0, 20, True), (2, 2, 80, False)], [(3, 3, 1, False)], time=10)
    memory = ARCHIVE_AGENT.initial_memory(obs.armies.shape)
    memory = memory._replace(
        last_turn=jnp.int32(9),
        base=memory.base._replace(
            last_turn=jnp.int32(9), tree=memory.base.tree._replace(phase=jnp.int32(1), last_turn=jnp.int32(9))
        ),
    )
    parent = PARENT.step(obs, KEY, ordinary_memory(memory))
    action, returned, tel = ARCHIVE_AGENT.step(obs, KEY, memory)
    assert tel["tree_urgent_priority"] and not tel["tree_action_issued"]
    assert returned.base.tree.phase == 0
    same_tree((action, ordinary_memory(returned)), parent[:2])
    if winning_general:
        np.testing.assert_array_equal(action, [0, 2, 2, 3, 0])
    else:
        assert action[0] == 2

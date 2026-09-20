"""Regression tests for deterministic Expander frontier selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


AGENT_PATH = Path(__file__).resolve().parents[1] / "competition/agents/expander_python/agent.py"
SPEC = importlib.util.spec_from_file_location("competition_expander_agent", AGENT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def observation(*, owned, armies, fog=(), enemies=(), turn=0, build_costs=None):
    height = width = 5
    type_grid = [[1 for _ in range(width)] for _ in range(height)]
    owner_grid = [[0 for _ in range(width)] for _ in range(height)]
    army_grid = [[0 for _ in range(width)] for _ in range(height)]
    for row, col in fog:
        type_grid[row][col] = 0
    for row, col in owned:
        owner_grid[row][col] = 1
    for row, col in enemies:
        owner_grid[row][col] = 2
    for (row, col), army in armies.items():
        army_grid[row][col] = army
    values = dict(
        H=height,
        W=width,
        turn=turn,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )
    if build_costs is not None:
        costs = [[0 for _ in range(width)] for _ in range(height)]
        for (row, col), cost in build_costs.items():
            costs[row][col] = cost
        values["build_cost_grid"] = costs
    return SimpleNamespace(**values)


def test_fog_capture_beats_friendly_north_transfer():
    obs = observation(
        owned={(1, 2), (0, 2), (2, 2), (1, 1)},
        armies={(1, 2): 8},
        fog={(1, 3)},
    )

    move = MODULE.Agent(0, obs.H, obs.W).act(obs)

    assert move == (0, 1, 2, 3, 0), "fog should count as expansion"


def test_equal_expansions_prefer_interior_over_north_wall():
    obs = observation(
        owned={(1, 2)},
        armies={(1, 2): 8},
        fog={(0, 2), (2, 2)},
    )

    move = MODULE.Agent(0, obs.H, obs.W).act(obs)

    assert move == (0, 1, 2, 1, 0), "equal frontier moves should move away from the edge"


def test_single_army_capture_beats_large_friendly_shuffle():
    obs = observation(owned={(1, 1), (1, 2), (3, 3)}, armies={(1, 1): 200, (1, 2): 1, (3, 3): 2})
    for _, (dr, dc) in enumerate(MODULE.DIRECTIONS):
        r, c = 1 + dr, 1 + dc
        if (r, c) != (1, 2):
            obs.type_grid[r][c] = 2
    move = MODULE.Agent(0, 5, 5).act(obs)
    assert move[1:3] == (3, 3)
    assert move[0] == 0


def test_nearby_surplus_gathers_and_captures_city():
    obs = observation(owned={(2, 1), (2, 2)}, armies={(2, 1): 30, (2, 2): 25, (2, 3): 45})
    obs.type_grid[2][1] = 4
    obs.type_grid[2][3] = 3
    agent = MODULE.Agent(0, 5, 5)
    assert agent.act(obs) == (0, 2, 1, 3, 0)
    obs.army_grid[2][1], obs.army_grid[2][2] = 1, 54
    assert agent.act(obs) == (0, 2, 2, 3, 0)


def test_enclosed_army_passes_instead_of_oscillating():
    obs = observation(owned={(2, 1), (2, 2)}, armies={(2, 1): 50, (2, 2): 2})
    for r in range(5):
        for c in range(5):
            if obs.owner_grid[r][c] != 1:
                obs.type_grid[r][c] = 2
    assert MODULE.Agent(0, 5, 5).act(obs) == MODULE.PASS


def test_city_gathering_advances_large_stack_before_new_capital_growth():
    obs = observation(owned={(2, 0), (2, 1), (2, 2)}, armies={(2, 0): 2, (2, 1): 60, (2, 2): 1, (2, 3): 45})
    obs.type_grid[2][0] = 4
    obs.type_grid[2][3] = 3
    assert MODULE.Agent(0, 5, 5).act(obs) == (0, 2, 1, 3, 0)


def test_understrength_siege_holds_front_and_feeds_it():
    obs = observation(
        owned={(2, 0), (2, 1), (2, 2)},
        enemies={(2, 3)},
        armies={(2, 0): 60, (2, 1): 2, (2, 2): 20, (2, 3): 45},
        turn=900,
    )
    obs.type_grid[2][3] = 4
    agent = MODULE.Agent(0, 5, 5)

    assert agent.act(obs) == (0, 2, 0, 3, 0)
    assert agent.enemy_general == (2, 3)
    assert agent.spearhead == (2, 2)


def test_siege_captures_general_as_soon_as_stack_is_ready():
    obs = observation(
        owned={(2, 1), (2, 2)},
        enemies={(2, 3)},
        armies={(2, 1): 10, (2, 2): 48, (2, 3): 45},
        turn=900,
    )
    obs.type_grid[2][3] = 4

    assert MODULE.Agent(0, 5, 5).act(obs) == (0, 2, 2, 3, 0)


def test_remembered_general_remains_the_objective_after_fog_returns():
    visible = observation(
        owned={(2, 0), (2, 1), (2, 2)},
        enemies={(2, 3)},
        armies={(2, 0): 50, (2, 1): 2, (2, 2): 10, (2, 3): 40},
        turn=500,
    )
    visible.type_grid[2][3] = 4
    agent = MODULE.Agent(0, 5, 5)
    agent.act(visible)

    hidden = observation(
        owned={(2, 0), (2, 1)},
        armies={(2, 0): 50, (2, 1): 2},
        fog={(2, 2)},
        turn=501,
    )
    hidden.type_grid[2][3] = 5
    assert agent.act(hidden) == (0, 2, 1, 3, 0)
    assert agent.enemy_general == (2, 3)
    assert agent.spearhead == (2, 2)


def test_captured_ffa_general_is_retired_as_siege_target():
    obs = observation(
        owned={(2, 1), (2, 2)}, armies={(2, 1): 20, (2, 2): 40}, turn=700,
    )
    obs.type_grid[2][2] = 3
    obs.players = ["Red", "Blue", "Green", "Purple"]
    agent = MODULE.Agent(0, 5, 5)
    agent.enemy_general = (2, 2)
    agent.spearhead = (2, 1)
    agent.pressure_anchor = (2, 1)

    assert agent.siege(obs, {(2, 1), (2, 2)}) is None
    assert agent.enemy_general is None
    assert agent.spearhead is None
    assert agent.pressure_anchor is None


def test_combat_uses_one_persistent_large_spearhead():
    obs = observation(
        owned={(1, 1), (1, 2), (3, 2)},
        enemies={(1, 3), (3, 3)},
        armies={(1, 1): 2, (1, 2): 40, (1, 3): 12, (3, 2): 4, (3, 3): 1},
        turn=900,
    )
    agent = MODULE.Agent(0, 5, 5)

    assert agent.act(obs) == (0, 1, 2, 3, 0)
    assert agent.spearhead == (1, 3)


def test_blocked_spearhead_gathers_instead_of_taking_cheap_border_tile():
    obs = observation(
        owned={(1, 0), (1, 1), (1, 2), (3, 2)},
        enemies={(1, 3), (3, 3)},
        armies={(1, 0): 60, (1, 1): 2, (1, 2): 10, (1, 3): 30, (3, 2): 3, (3, 3): 1},
        turn=900,
    )
    agent = MODULE.Agent(0, 5, 5)
    agent.spearhead = (1, 2)

    assert agent.act(obs) == (0, 1, 0, 3, 0)
    assert agent.spearhead == (1, 2)


def test_late_explore_replaces_an_interior_spearhead():
    obs = observation(
        owned={(0, 2), (1, 1), (1, 2), (1, 3), (2, 2), (3, 3)},
        armies={(0, 2): 1, (1, 1): 60, (1, 2): 2, (1, 3): 1, (2, 2): 1, (3, 3): 20},
        turn=900,
    )
    agent = MODULE.Agent(0, 5, 5)
    agent.spearhead = (1, 2)
    obs.players = ["Red", "Blue", "Green", "Purple"]

    move = agent.act(obs)

    assert move[:3] == (0, 1, 1)
    assert agent.spearhead != (1, 2)


def test_ffa_pressure_keeps_one_rally_point_instead_of_switching_borders():
    obs = observation(
        owned={(1, 0), (1, 1), (1, 2), (3, 1), (3, 2)},
        enemies={(1, 3), (3, 3)},
        armies={(1, 0): 60, (1, 1): 2, (1, 2): 10, (1, 3): 30,
                (3, 1): 45, (3, 2): 12, (3, 3): 30},
        turn=900,
    )
    obs.players = ["Red", "Blue", "Green", "Purple"]
    agent = MODULE.Agent(0, 5, 5)
    agent.pressure_anchor = (1, 2)

    assert agent.act(obs) == (0, 1, 0, 3, 0)
    assert agent.pressure_anchor == (1, 2)


def test_ffa_rallies_to_exposed_general_before_distant_siege():
    obs = observation(
        owned={(2, 1), (2, 2), (2, 3), (3, 2)},
        enemies={(0, 2), (4, 3)},
        armies={(2, 1): 22, (2, 2): 5, (2, 3): 2, (3, 2): 18,
                (0, 2): 41, (4, 3): 1},
        turn=320,
    )
    obs.type_grid[2][2] = 4
    obs.type_grid[4][3] = 4
    obs.players = ["Red", "Blue", "Green", "Purple"]
    agent = MODULE.Agent(0, 5, 5)

    assert agent.act(obs) == (0, 2, 1, 3, 0)
    assert agent.enemy_general == (4, 3)


def test_ffa_does_not_rally_for_weak_or_distant_enemy():
    obs = observation(
        owned={(2, 1), (2, 2), (3, 2)}, enemies={(0, 2)},
        armies={(2, 1): 22, (2, 2): 5, (3, 2): 18, (0, 2): 1},
        turn=320,
    )
    obs.type_grid[2][2] = 4
    obs.players = ["Red", "Blue", "Green", "Purple"]

    assert MODULE.Agent(0, 5, 5).ffa_home_defense(
        obs, [(2, 1), (2, 2), (3, 2)], [(0, 2)]
    ) is None


def test_ffa_preserves_opening_against_early_contact():
    obs = observation(
        owned={(2, 1), (2, 2), (3, 2)}, enemies={(0, 2)},
        armies={(2, 1): 22, (2, 2): 5, (3, 2): 18, (0, 2): 100},
        turn=120,
    )
    obs.type_grid[2][2] = 4
    obs.players = ["Red", "Blue", "Green", "Purple"]

    assert MODULE.Agent(0, 5, 5).ffa_home_defense(
        obs, [(2, 1), (2, 2), (3, 2)], [(0, 2)]
    ) is None


def test_ffa_ignores_moderate_non_imminent_scout():
    obs = observation(
        owned={(2, 1), (2, 2), (3, 2)}, enemies={(0, 0)},
        armies={(2, 1): 22, (2, 2): 5, (3, 2): 18, (0, 0): 12},
        turn=320,
    )
    obs.type_grid[2][2] = 4
    obs.players = ["Red", "Blue", "Green", "Purple"]

    assert MODULE.Agent(0, 5, 5).ffa_home_defense(
        obs, [(2, 1), (2, 2), (3, 2)], [(0, 0)]
    ) is None


def test_classic_keeps_original_late_capture_order():
    obs = observation(
        owned={(1, 1), (1, 2), (3, 3)},
        armies={(1, 1): 50, (1, 2): 2, (3, 3): 3},
        turn=900,
    )
    agent = MODULE.Agent(0, 5, 5)
    agent.spearhead = (1, 1)

    assert agent.act(obs)[1:3] == (1, 2)


def test_castle_variant_funds_and_builds_one_opening_castle():
    waiting = observation(
        owned={(2, 2)}, armies={(2, 2): 49}, turn=96, build_costs={}
    )
    waiting.type_grid[2][2] = 4
    agent = MODULE.Agent(0, 5, 5)
    assert agent.act(waiting) == MODULE.PASS

    funded = observation(
        owned={(2, 2)}, armies={(2, 2): 50}, turn=98, build_costs={}
    )
    funded.type_grid[2][2] = 4
    assert agent.act(funded)[0:3] == (0, 2, 2)

    affordable = observation(
        owned={(2, 2), (2, 3)},
        armies={(2, 2): 1, (2, 3): 49},
        turn=99,
        build_costs={(2, 3): 47},
    )
    affordable.type_grid[2][2] = 4
    assert agent.act(affordable) == (2, 2, 3, 0, 0)

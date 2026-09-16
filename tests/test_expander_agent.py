"""Regression tests for deterministic Expander frontier selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


AGENT_PATH = Path(__file__).resolve().parents[1] / "competition/agents/expander_python/agent.py"
SPEC = importlib.util.spec_from_file_location("competition_expander_agent", AGENT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def observation(*, owned, armies, fog=(), enemies=(), turn=0):
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
    return SimpleNamespace(
        H=height,
        W=width,
        turn=turn,
        type_grid=type_grid,
        owner_grid=owner_grid,
        army_grid=army_grid,
    )


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

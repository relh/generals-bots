"""Regression tests for deterministic Expander frontier selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


AGENT_PATH = Path(__file__).resolve().parents[1] / "competition/agents/expander_python/agent.py"
SPEC = importlib.util.spec_from_file_location("competition_expander_agent", AGENT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def observation(*, owned, armies, fog=()):
    height = width = 5
    type_grid = [[1 for _ in range(width)] for _ in range(height)]
    owner_grid = [[0 for _ in range(width)] for _ in range(height)]
    army_grid = [[0 for _ in range(width)] for _ in range(height)]
    for row, col in fog:
        type_grid[row][col] = 0
    for row, col in owned:
        owner_grid[row][col] = 1
    for (row, col), army in armies.items():
        army_grid[row][col] = army
    return SimpleNamespace(
        H=height,
        W=width,
        turn=0,
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

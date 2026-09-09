"""Historical replay must preserve stochastic keys, seats and arena termination."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents import RandomAgent
from generals.evaluation import replay
from generals.evaluation.arena import Rules, make_runner
from generals.evaluation.replay import compare_result, load_grid, replay_game, select_rows


@pytest.mark.parametrize("seat", [0, 1])
def test_replay_matches_batched_stochastic_arena_and_counter_order(seat):
    grid = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int32)
    candidate, opponent = RandomAgent().act, RandomAgent().act
    rules = Rules(max_turns=40)
    expected = make_runner(candidate, opponent, rules)(
        jnp.array(grid[None]), jax.random.PRNGKey(183)[None], jnp.array([seat])
    )
    arrays, actual = replay_game(grid, candidate, opponent, rules, seat=seat, action_seed=183)
    assert actual["turns"] == int(expected.state.time[0])
    assert actual["winner"] == int(expected.state.winner[0])
    assert actual["terminal"] == bool(expected.finished[0])
    np.testing.assert_array_equal(arrays["action_counters"].sum(axis=0), expected.counters[0])
    for field in expected.state._fields:
        np.testing.assert_array_equal(arrays[f"state_{field}"][-1], getattr(expected.state, field)[0])
    assert len(arrays["actions"]) + 1 == len(arrays["state_time"])
    for player in (0, 1):
        fog = arrays[f"observation_{player}_fog_cells"]
        assert np.all(arrays[f"observation_{player}_armies"][fog] == 0)
    assert compare_result(actual, {key: str(value) for key, value in actual.items()}) == {}
    assert "turns" in compare_result(actual, {"turns": actual["turns"] + 1})


def test_load_stored_board_swaps_only_general_labels(tmp_path):
    grid = np.array([[1, -1, 2], [0, 25, 0]], np.int32)
    np.savez_compressed(tmp_path / "open4_boards.npz", **{"9": grid})
    row = dict(suite="open4", board_id="9", swapped="1", height="2", width="3")
    np.testing.assert_array_equal(load_grid(tmp_path, row), [[2, -1, 1], [0, 25, 0]])


def test_representative_selection_cycles_opponent_groups():
    rows = [
        dict(suite="open4", opponent=opponent, board_id=str(i), result="loss")
        for opponent in ("hunter", "random")
        for i in range(3)
    ]
    assert [r["opponent"] for r in select_rows(rows, max_samples=2)] == ["hunter", "random"]
    assert select_rows(rows, result="win") == []


def test_snapshot_provenance_exempts_current_candidate_but_not_core_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    agent_path = tmp_path / "generals/agents/sentinel_agent.py"
    core_path = tmp_path / "generals/core/game.py"
    agent_path.parent.mkdir(parents=True)
    core_path.parent.mkdir(parents=True)
    agent_path.write_text("current agent")
    core_path.write_text("new core")
    snapshot = tmp_path / "snapshot.py"
    snapshot.write_text("historical agent")
    metadata = dict(
        source_hashes={"generals/agents/sentinel_agent.py": "old", "generals/core/game.py": "old"},
        candidate_source_sha256=replay.file_hash(snapshot),
    )
    provenance = replay.source_provenance(metadata, dict(candidate="sentinel", opponent="hunter"), snapshot)
    assert provenance["critical_changed_sources"] == ["generals/core/game.py"]
    assert provenance["candidate_source_sha256"] == replay.file_hash(snapshot)


@pytest.mark.parametrize("alias", ["sentinel-v3", "sentinel-v3-memory", "sentinel-v3-defense", "sentinel-v3-disabled"])
def test_v3_snapshot_guard_checks_base_dependency_cli_and_snapshot_identity(tmp_path, monkeypatch, alias):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    names = ["generals/agents/sentinel_v3_agent.py", "generals/agents/sentinel_agent.py", "generals/evaluation/cli.py"]
    recorded = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("historical " + name)
        recorded[name] = replay.file_hash(path)
    snapshot = tmp_path / "v3_snapshot.py"
    snapshot.write_bytes((tmp_path / names[0]).read_bytes())
    for name in names:
        (tmp_path / name).write_text("changed " + name)
    provenance = replay.source_provenance(
        dict(source_hashes=recorded), dict(candidate=alias, opponent="hunter"), snapshot
    )
    assert set(provenance["critical_changed_sources"]) == set(names[1:])
    snapshot.write_text("wrong snapshot")
    provenance = replay.source_provenance(
        dict(source_hashes=recorded), dict(candidate=alias, opponent="hunter"), snapshot
    )
    assert "candidate_source_sha256" in provenance["critical_changed_sources"]


@pytest.mark.parametrize("version", [4, 5])
def test_stateless_snapshot_guard_checks_both_frozen_scoring_dependencies(tmp_path, monkeypatch, version):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    names = [
        f"generals/agents/sentinel_v{version}_agent.py",
        "generals/agents/sentinel_v3_agent.py",
        "generals/agents/sentinel_agent.py",
    ]
    recorded = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("historical " + name)
        recorded[name] = replay.file_hash(path)
    snapshot = tmp_path / f"v{version}_snapshot.py"
    snapshot.write_bytes((tmp_path / names[0]).read_bytes())
    for name in names:
        (tmp_path / name).write_text("changed " + name)
    provenance = replay.source_provenance(
        dict(source_hashes=recorded), dict(candidate=f"sentinel-v{version}", opponent="hunter"), snapshot
    )
    assert set(provenance["critical_changed_sources"]) == set(names[1:])

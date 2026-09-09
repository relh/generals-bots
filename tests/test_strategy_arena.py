"""Strategy-only adapter preserves policy history and never invents timed-out actions."""

import json
import sys
import zipfile
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from generals.evaluation.arena import Rules
from scripts import strategy_arena as arena


@pytest.fixture
def external(tmp_path, monkeypatch):
    directory = tmp_path / "external"
    directory.mkdir()
    sources = {
        "numpy_infer.py": "OFFSET=10\n",
        "agent.py": """from numpy_infer import OFFSET
class Agent:
    def __init__(self, player_id, H, W):
        self.player = player_id
        self.count = 0
    def act(self, obs):
        self.count += 1
        return [0, self.player, obs + OFFSET, self.count, 0]
""",
        "main.py": """import sys
from agent import Agent
def _read_observation(stdin,H,W,line):
    return int(line)
if __name__ == '__main__':
    player,H,W=map(int,sys.stdin.readline().split())
    agent=Agent(player,H,W)
    for line in sys.stdin:
        print(*agent.act(_read_observation(sys.stdin,H,W,line)),flush=True)
""",
        "run.sh": "#!/bin/sh\nexec python -u main.py\n",
    }
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, source in sources.items():
            (directory / name).write_text(source)
            bundle.writestr(name, source)
    monkeypatch.setattr(arena, "ARCHIVE_SHA", arena.sha(archive.read_bytes()))
    return directory, archive


def test_identity_rejects_changed_archive_before_import(external):
    directory, archive = external
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="reviewed pinned"):
        arena.Amin(directory, archive)


@pytest.mark.parametrize("change", ["source", "extra", "symlink"])
def test_identity_rejects_directory_repairs(external, change):
    directory, archive = external
    if change == "source":
        (directory / "agent.py").write_text("raise RuntimeError('must not execute')")
    elif change == "extra":
        (directory / "repair.py").write_text("")
    else:
        original = directory / "agent.py"
        moved = directory.parent / "agent.py"
        original.rename(moved)
        original.symlink_to(moved)
    with pytest.raises(ValueError, match="differs"):
        arena.Amin(directory, archive)


def test_original_parser_memory_and_import_namespace(external, monkeypatch):
    original = SimpleNamespace(unrelated=True)
    monkeypatch.setitem(sys.modules, "agent", original)
    amin = arena.Amin(*external)
    assert sys.modules["agent"] is original
    first = amin.new_game(1, (18, 21))
    assert amin.act(first, "3\n", (18, 21)) == [0, 1, 13, 1, 0]
    assert amin.act(first, "8\n", (18, 21)) == [0, 1, 18, 2, 0]
    second = amin.new_game(1, (18, 21))
    assert amin.act(second, "3\n", (18, 21)) == [0, 1, 13, 1, 0]
    with pytest.raises(ValueError, match="Trailing"):
        amin.act(first, "3\n4\n", (18, 21))


def test_full_history_stdio_parity_and_recorded_original(external):
    amin = arena.Amin(*external)
    history = {
        "shape": [18, 21],
        "player_id": 1,
        "observations": ["3\n", "8\n"],
        "expected_original_raw_actions": [[0, 1, 13, 1, 0], [0, 1, 18, 2, 0]],
    }
    result = arena.parity_history(amin, history, timeout=5)
    assert result["exact_action_parity"] and result["recorded_actions_checked"]
    assert result["frames"] == 2
    history["expected_original_raw_actions"][1][3] = 1
    with pytest.raises(ValueError, match="recorded original"):
        arena.parity_history(amin, history, timeout=5)


def test_parity_does_not_hide_stdout_mismatch(external, monkeypatch):
    amin = arena.Amin(*external)
    monkeypatch.setattr(amin, "act", lambda *args: [1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="history mismatch"):
        arena.parity_history(amin, {"shape": [18, 21], "player_id": 0, "observations": ["3\n"]}, timeout=5)


def test_timing_never_replaces_slow_action_or_rounds_values(monkeypatch):
    wall, cpu = iter([0, 20_000_000_000]), iter([0, 100_000_000])
    monkeypatch.setattr(arena.time, "perf_counter_ns", lambda: next(wall))
    monkeypatch.setattr(arena.time, "process_time_ns", lambda: next(cpu))
    raw, timing = arena.timed(lambda: np.array([0, 1, 2, 3, 0.5]))
    assert raw == [0, 1, 2, 3, 0.5]
    assert timing == {"wall_seconds": 20, "process_cpu_seconds": 0.1}


class PassPolicy:
    def act(self, obs, key):
        return jnp.array([1, 0, 0, 0, 0])


class PassAmin:
    def __init__(self):
        self.instances = []

    def new_game(self, player, shape):
        self.instances.append({"calls": 0})
        return self.instances[-1]

    def act(self, instance, wire, shape):
        assert len(wire.splitlines()) == 1 + 3 * shape[0]
        instance["calls"] += 1
        return [1, 0, 0, 0, 0]


def tiny_case(game_id=0):
    return dict(
        grid=np.array([[1, 0, 0], [0, 0, 2]], dtype=np.int32),
        seat=0,
        board_id=0,
        swapped=0,
        repeat=0,
        action_seed=123,
        game_id=game_id,
    )


def test_real_engine_turns_full_trace_and_per_game_reset(tmp_path):
    rules, amin, observe = Rules(max_turns=2), PassAmin(), arena.engine()
    arena.warm_engine(observe, tiny_case()["grid"], rules)
    policy = PassPolicy()
    for index in range(2):
        row = arena.play_case(tiny_case(index), rules, "sentinel", amin, tmp_path, policy, observe)
        assert row["turns"] == 2
        assert row["result"] == "draw" and row["score"] == 0.5
        assert "action_seed" not in row
    assert len(amin.instances) == 2 and all(x["calls"] == 2 for x in amin.instances)
    trace = json.loads((tmp_path / "game-0000.json").read_text())
    assert len(trace["frames"]) == 2 and trace["error"] is None
    for frame in trace["frames"]:
        for player in frame["players"]:
            assert player["raw_action"] == player["applied_action"] == [1, 0, 0, 0, 0]
            assert player["wall_seconds"] >= 0 and player["process_cpu_seconds"] >= 0
            assert player["wire_observation"] and not player["invalid_action"]


def test_infrastructure_error_saves_partial_without_pass_or_result(tmp_path):
    class Broken(PassAmin):
        def act(self, *args):
            raise RuntimeError("broken adapter")

    with pytest.raises(RuntimeError, match="broken adapter"):
        arena.play_case(tiny_case(), Rules(max_turns=2), "sentinel", Broken(), tmp_path, PassPolicy(), arena.engine())
    trace = json.loads((tmp_path / "game-0000.json").read_text())
    assert trace["game"]["result"] == "unfinished" and trace["game"]["score"] is None
    assert trace["game"]["reason"] == "infrastructure_error"
    assert "applied_action" not in trace["frames"][0]["players"][1]


def test_runner_invalid_action_semantics_preserve_raw(tmp_path):
    class Invalid(PassAmin):
        def act(self, *args):
            return [0, 999, 999, 0, 0]

    arena.play_case(tiny_case(), Rules(max_turns=1), "sentinel", Invalid(), tmp_path, PassPolicy(), arena.engine())
    reply = json.loads((tmp_path / "game-0000.json").read_text())["frames"][0]["players"][1]
    assert reply["raw_action"] == [0, 999, 999, 0, 0]
    assert reply["applied_action"] == [1, 0, 0, 0, 0]
    assert reply["invalid_action"] and not reply["malformed_command"]


def test_output_refuses_overwrite_before_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "strategy_arena.py",
            "run",
            "--external-directory",
            ".",
            "--external-archive",
            "x.zip",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        arena.main()


def test_candidate_memory_and_rng_reset_with_reused_policy(tmp_path):
    class Stateful:
        def __init__(self):
            self.initialized = 0
            self.keys = []

        def initial_memory(self, shape):
            self.initialized += 1
            return 0

        def step(self, observation, key, memory):
            self.keys.append(np.asarray(key).tolist())
            return jnp.array([1, memory, 0, 0, 0]), memory + 1, {}

    policy, observe = Stateful(), arena.engine()
    for index in range(2):
        arena.play_case(tiny_case(index), Rules(max_turns=2), "sentinel", PassAmin(), tmp_path, policy, observe)
        trace = json.loads((tmp_path / f"game-{index:04d}.json").read_text())
        assert [f["players"][0]["raw_action"][1] for f in trace["frames"]] == [0, 1]
    assert policy.initialized == 2
    assert policy.keys[:2] == policy.keys[2:]

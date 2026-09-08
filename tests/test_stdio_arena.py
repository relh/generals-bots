"""Synthetic subprocess regressions for deadline ownership and fault accounting."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/stdio_arena.py"
spec = importlib.util.spec_from_file_location("stdio_arena", SCRIPT)
arena = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = arena
spec.loader.exec_module(arena)


def start(tmp_path, code, limits=None):
    return arena.AgentProcess(
        [sys.executable, "-u", "-c", code],
        tmp_path,
        os.environ.copy(),
        tmp_path / "stderr.log",
        limits=limits or arena.Limits(first_seconds=1),
    )


def request(agent, payload=b"frame\n"):
    agent.begin(payload, (3, 3))
    while agent.poll() is None:
        time.sleep(0.001)
    return agent.result


def test_fifty_malformed_responses_exhaust_fault_budget(tmp_path):
    agent = start(tmp_path, "import sys\nfor line in sys.stdin: print('bad command',flush=True)")
    try:
        for _ in range(50):
            result = request(agent)
            assert result["action"] == arena.PASS
            assert result["fault"] == "malformed"
        assert agent.faults == agent.limits.fault_budget == 50
        assert agent.fault_reasons == {"malformed": 50}
    finally:
        agent.close()


def test_late_reply_is_never_credited_to_next_frame(tmp_path):
    agent = start(
        tmp_path,
        """import sys,time
for index,line in enumerate(sys.stdin):
    if index == 1:
        time.sleep(.16)
        print('0 0 0 3 0',flush=True)
    else:
        print('2 1 1 0 0',flush=True)
""",
        arena.Limits(first_seconds=1, turn_seconds=0.04),
    )
    try:
        assert request(agent)["action"] == [2, 1, 1, 0, 0]
        missed = request(agent)
        assert missed["fault"] == "deadline"
        assert missed["action"] == arena.PASS
        skipped = []
        for _ in range(10):
            if not agent.pending:
                break
            result = request(agent)
            skipped.append(result)
            assert result["action"] == arena.PASS
            assert result["fault"] == "pending_late_reply"
        assert skipped
        assert not agent.pending
        assert agent.stale_lines == 1
        assert request(agent)["action"] == [2, 1, 1, 0, 0]
    finally:
        agent.close()


def test_crash_forfeits_immediately_without_waiting_for_fault_budget(tmp_path):
    agent = start(tmp_path, "import sys\nsys.stdin.readline()\nsys.exit(9)")
    try:
        request(agent)
        deadline = time.monotonic() + 1
        while agent.forfeit_reason is None and time.monotonic() < deadline:
            agent.poll()
            time.sleep(0.001)
        assert agent.forfeit_reason == "process_exit"
        assert agent.faults < 50
    finally:
        agent.close()
    assert agent.process.returncode == 9


def test_exit_is_bounded_for_unresponsive_process(tmp_path):
    agent = start(tmp_path, "import time\ntime.sleep(60)", arena.Limits(exit_seconds=0.05))
    started = time.monotonic()
    agent.close()
    assert time.monotonic() - started < 2
    assert agent.process.poll() is not None


@pytest.mark.parametrize("line", [b"0 -1 0 2 0", b"0 0 0 9 0", b"4 0 0 0 0", b"0 0 0 0 2", b"1 0.0 0 0 0"])
def test_invalid_commands_become_pass(line):
    action, fault = arena.parse_action(line, (3, 3))
    assert action == arena.PASS
    assert fault


def test_correctly_formatted_illegal_actions_do_not_add_faults(tmp_path):
    agent = start(tmp_path, "import sys\nfor line in sys.stdin: print('0 -1 0 3 0',flush=True)")
    try:
        for _ in range(50):
            result = request(agent)
            assert result["action"] == arena.PASS
            assert result["fault"] is None
        assert agent.faults == 0
        assert agent.invalid_actions == 50
    finally:
        agent.close()


def test_physically_invalid_move_is_silent_pass_without_runner_fault(tmp_path):
    agent = start(tmp_path, "import sys\nfor line in sys.stdin: print('0 0 0 0 0',flush=True)")
    try:
        for _ in range(50):
            request(agent)  # UP from the top row is syntactically valid but impossible.
            agent.replace_invalid_action()
            assert agent.result["action"] == arena.PASS
            assert agent.result["fault"] is None
        assert agent.faults == 0
        assert agent.invalid_actions == 50
    finally:
        agent.close()


def test_early_exit_does_not_wait_for_opponents_later_exit(tmp_path):
    first = start(tmp_path, "import sys,time\nsys.stdin.readline()\ntime.sleep(.03)\nsys.exit(9)")
    second = start(tmp_path, "import sys,time\nsys.stdin.readline()\ntime.sleep(5)\nsys.exit(9)")
    try:
        started = time.monotonic()
        arena.ask_pair([first, second], [b"frame\n", b"frame\n"], (3, 3))
        assert time.monotonic() - started < 1
        assert first.forfeit_reason == "process_exit"
        assert second.forfeit_reason is None
        assert second.process.poll() is None
    finally:
        first.close()
        second.close()


def test_partial_late_line_is_drained_as_one_old_response(tmp_path):
    agent = start(
        tmp_path,
        """import sys,time
for index,line in enumerate(sys.stdin):
    if index == 1:
        sys.stdout.write('0 0 ');sys.stdout.flush();time.sleep(.12)
        print('0 3 0',flush=True)
    else:
        print('1 0 0 0 0',flush=True)
""",
        arena.Limits(first_seconds=1, turn_seconds=0.03),
    )
    try:
        assert request(agent)["fault"] is None
        assert request(agent)["fault"] == "deadline"
        for _ in range(10):
            if not agent.pending:
                break
            assert request(agent)["action"] == arena.PASS
        assert agent.stale_lines == 1
        assert not agent.pending
        assert request(agent)["fault"] is None
    finally:
        agent.close()


def test_final_kill_reap_timeout_is_bounded_and_closes_every_descriptor():
    calls = []

    def wait(timeout):
        calls.append(timeout)
        raise subprocess.TimeoutExpired("synthetic", timeout)

    agent = object.__new__(arena.AgentProcess)
    agent.limits = arena.Limits(exit_seconds=0.01)
    agent.stderr = io.BytesIO()
    agent.process = SimpleNamespace(
        pid=987654321, stdin=io.BytesIO(), stdout=io.BytesIO(), returncode=None, wait=wait, poll=lambda: None
    )
    signals = []
    agent._kill_group = signals.append
    report = agent.close()
    assert calls == [0.01, 0.01, 0.01]
    assert report["final_reap_timeout"] and not report["reaped"]
    assert [step["stage"] for step in report["waits"]] == ["eof", "term", "kill"]
    assert len(signals) == 3
    assert agent.process.stdin.closed and agent.process.stdout.closed and agent.stderr.closed


def test_cleanup_attempts_second_agent_after_first_cleanup_raises():
    attempted = []

    def first():
        attempted.append(1)
        raise RuntimeError("cleanup exploded")

    def second():
        attempted.append(2)
        return dict(reaped=True)

    reports = arena.close_all(
        [
            SimpleNamespace(process=SimpleNamespace(pid=1), close=first),
            SimpleNamespace(process=SimpleNamespace(pid=2), close=second),
        ]
    )
    assert attempted == [1, 2]
    assert not reports[0]["reaped"] and reports[1]["reaped"]


@pytest.mark.parametrize("reporting_failure", [False, True])
def test_cleanup_failure_preserves_finished_game_before_attempting_both_agents(
    tmp_path, monkeypatch, reporting_failure
):
    import numpy as np

    from generals.evaluation.arena import Rules

    attempts = []
    turns = 2 if reporting_failure else 1
    if reporting_failure:

        def interrupted(values):
            raise KeyboardInterrupt("synthetic statistics interruption")

        monkeypatch.setattr(np, "median", interrupted)

    class FakeAgent:
        def __init__(self, *args):
            self.index = len(created)
            created.append(self)
            self.process = SimpleNamespace(pid=987654320 + self.index, returncode=None)
            self.faults = self.invalid_actions = self.peak_sampled_tree_rss_bytes = self.peak_sampled_rss_bytes = 0
            self.fault_reasons = {}
            self.stale_lines = 0
            self.forfeit_reason = None
            self.limits = arena.Limits()

        def close(self):
            saved = json.loads((tmp_path / "game-0000.json").read_text())
            assert saved["game"]["result"] == "draw" and len(saved["frames"]) == turns
            assert saved["cleanup"]["status"] == "pending"
            attempts.append(self.index)
            self.process.returncode = 0 if reporting_failure or self.index == 1 else None
            return dict(
                reaped=reporting_failure or self.index == 1,
                final_reap_timeout=not reporting_failure and self.index == 0,
            )

    created = []
    monkeypatch.setattr(arena, "AgentProcess", FakeAgent)
    monkeypatch.setattr(
        arena,
        "ask_pair",
        lambda *args: [dict(action=arena.PASS.copy(), fault=None, response_seconds=0.01) for _ in range(2)],
    )
    with pytest.raises(arena.CaseAborted) as error:
        arena.play_case(
            [tmp_path / "a", tmp_path / "b"],
            np.array([[1, 0], [0, 2]], np.int32),
            dict(game_id=0, seat=0),
            Rules(max_turns=turns),
            tmp_path,
            {},
            [0, 1],
        )
    assert attempts == [0, 1]
    assert error.value.row["result"] == "draw" and error.value.row["turns"] == turns
    saved = json.loads((tmp_path / "game-0000.json").read_text())
    assert saved["cleanup"]["status"] == ("complete" if reporting_failure else "failed")
    assert saved["cleanup"]["processes"][0]["final_reap_timeout"] == (not reporting_failure)
    if reporting_failure:
        assert "KeyboardInterrupt" in saved["runner_error"]
    assert len(saved["cleanup"]["processes"]) == 2


def complete_details(row):
    return dict(
        game=row,
        frames=[
            dict(
                turn=0,
                faults=[0, 0],
                replies=[dict(action=arena.PASS.copy(), fault=None, response_seconds=0.01) for _ in range(2)],
            )
        ],
        processes=[dict(pid=987654320 + i, faults=0, fault_reasons={}) for i in range(2)],
        cleanup=dict(status="complete"),
    )


def completed_fixture(tmp_path):
    import csv

    row = dict(
        game_id=0,
        board_id=0,
        repeat=0,
        swapped=0,
        seat=0,
        height=2,
        width=2,
        turns=1,
        result="win",
        winner=0,
        reason="terminal",
        candidate_faults=0,
        opponent_faults=0,
    )
    arena.atomic_json(
        tmp_path / "game-0000.json",
        complete_details(row),
    )
    with (tmp_path / "games.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    case = {key: row[key] for key in ("game_id", "board_id", "repeat", "swapped", "seat")}
    return row, [case]


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "partial_csv",
        "partial_frames",
        "changed_case",
        "unfinished",
        "conflict",
        "missing_replies",
        "missing_processes",
        "wrong_faults",
    ],
)
def test_resume_rejects_partial_duplicate_or_conflicting_cases(tmp_path, mutation):
    row, cases = completed_fixture(tmp_path)
    if mutation == "duplicate":
        text = (tmp_path / "games.csv").read_text()
        (tmp_path / "games.csv").write_text(text + text.splitlines()[-1] + "\n")
    elif mutation == "partial_csv":
        with (tmp_path / "games.csv").open("a") as stream:
            stream.write("1,0\n")
    elif mutation == "changed_case":
        cases[0]["seat"] = 1
    else:
        path = tmp_path / "game-0000.json"
        details = json.loads(path.read_text())
        if mutation == "partial_frames":
            details["frames"] = []
        if mutation == "unfinished":
            details["game"]["result"] = "unfinished"
        if mutation == "conflict":
            details["game"]["winner"] = 1
        if mutation == "missing_replies":
            details["frames"][0].pop("replies")
        if mutation == "missing_processes":
            details["processes"] = []
        if mutation == "wrong_faults":
            details["frames"][0]["faults"] = [1, 0]
        arena.atomic_json(path, details)
    with pytest.raises(ValueError):
        arena.validate_completed_rows(tmp_path, cases)


def test_resume_recovers_only_complete_durable_json_without_rerunning(tmp_path):
    row, cases = completed_fixture(tmp_path)
    (tmp_path / "games.csv").unlink()
    rows, recovered = arena.validate_completed_rows(tmp_path, cases)
    assert rows == [row] and recovered == [0]


@pytest.mark.parametrize("field", ["candidate", "opponent", "runtime", "cpus", "seed", "rules", "boards_sha256"])
def test_resume_rejects_changed_policy_runtime_options_or_boards(field):
    old = dict(
        candidate={"directory_sha256": "policy"},
        opponent={"archive_sha256": "opponent"},
        runtime={"jax": "fixed"},
        cpus=[1, 2],
        seed=3,
        rules={"max_turns": 1200},
        boards_sha256="original",
        source_sha256={"scripts/stdio_arena.py": "same", "generals/core/game.py": "same"},
    )
    current = dict(old)
    current[field] = "changed"
    with pytest.raises(ValueError):
        arena.verify_resume_metadata(old, current)


def test_same_output_lock_refuses_live_runner(tmp_path):
    with arena.RunLock(tmp_path):
        with pytest.raises(ValueError, match="holds"):
            arena.RunLock(tmp_path)


def test_distinct_fresh_outputs_may_share_live_agent_path(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(30)"], cwd=agent_dir, start_new_session=True
    )
    try:
        paths = [agent_dir / "run.sh"] * 2
        arena.assert_no_live_processes(first, paths)
        arena.assert_no_live_processes(second, paths)
        arena.atomic_json(first / "children.json", [arena.process_record(child.pid)])
        with pytest.raises(ValueError, match="still present"):
            arena.assert_no_live_processes(first, paths)
        arena.assert_no_live_processes(second, paths)
        with pytest.raises(ValueError, match="Live runner"):
            arena.assert_no_live_processes(second, paths, legacy=True)
    finally:
        child.kill()
        child.wait(timeout=2)


def test_csv_preserved_before_cleanup_abort_and_resume_runs_only_missing_cases(tmp_path, monkeypatch):
    import csv

    import numpy as np

    from generals.core.env import GeneralsEnv
    from generals.evaluation import scenarios
    from generals.evaluation.arena import Rules

    env = GeneralsEnv(mode="competition")
    rules = Rules(env.truncation, env.build_castles, env.deathtouch_turn)
    monkeypatch.setattr(scenarios, "make_suite", lambda *args: ([np.array([[1, 0], [0, 2]], np.int32)], rules))
    args = SimpleNamespace(
        output=tmp_path,
        boards=1,
        seed=7,
        resume=False,
        resume_equivalence=None,
        check_resume=False,
        diagnostic_turns=None,
    )
    metadata = dict(
        source_sha256={"scripts/stdio_arena.py": arena.sha256(SCRIPT)},
        segments_required=True,
        schema_version=2,
        seed=7,
        boards=1,
        diagnostic_turns=None,
    )
    called = []

    def play(paths, grid, case, *args):
        called.append(case["game_id"])
        row = dict(
            case,
            height=2,
            width=2,
            turns=1,
            result="win",
            winner=case["seat"],
            reason="terminal",
            candidate_faults=0,
            opponent_faults=0,
        )
        arena.atomic_json(
            tmp_path / f"game-{case['game_id']:04d}.json",
            dict(complete_details(row), cleanup=dict(status="failed" if case["game_id"] == 0 else "complete")),
        )
        if case["game_id"] == 0:
            raise arena.CaseAborted("synthetic final reap timeout", row)
        return row

    monkeypatch.setattr(arena, "play_case", play)
    paths = [tmp_path / "agent1/run.sh", tmp_path / "agent2/run.sh"]
    with pytest.raises(arena.CaseAborted):
        arena.run_output(args, paths, {}, [2, 3], metadata)
    original = (tmp_path / "metadata.json").read_bytes()
    with (tmp_path / "games.csv").open() as stream:
        assert [int(row["game_id"]) for row in csv.DictReader(stream)] == [0]
    assert json.loads((tmp_path / "segments/0000.result.json").read_text())["status"] == "aborted"
    args.resume = True
    arena.run_output(args, paths, {}, [2, 3], dict(metadata))
    assert called == [0, 1, 2, 3]
    assert (tmp_path / "metadata.json").read_bytes() == original
    with (tmp_path / "games.csv").open() as stream:
        assert [int(row["game_id"]) for row in csv.DictReader(stream)] == [0, 1, 2, 3]
    segment = json.loads((tmp_path / "segments/0001.json").read_text())
    assert segment["completed_game_ids"] == [0] and segment["pending_game_ids"] == [1, 2, 3]
    assert segment["source_equivalence"] is None
    # A crash after the final durable row but before summary publication must
    # not require a rerun, and a zero-pending resume must repair the summary.
    (tmp_path / "summary.json").unlink()
    arena.run_output(args, paths, {}, [2, 3], dict(metadata))
    assert called == [0, 1, 2, 3]
    assert json.loads((tmp_path / "summary.json").read_text())["completed_games"] == 4


@pytest.mark.parametrize(
    "before,after",
    [
        ("turn_seconds: float = 0.150", "turn_seconds: float = 0.151"),
        ("PASS = [1, 0, 0, 0, 0]", "PASS = [0, 0, 0, 0, 0]"),
        ("play_error = None", "play_error = dangerous()"),
    ],
)
def test_cleanup_equivalence_rejects_changed_deadline_ast(tmp_path, monkeypatch, before, after):
    old_source = tmp_path / "old.py"
    old_source.write_text(SCRIPT.read_text())
    changed = tmp_path / "new.py"
    changed.write_text(SCRIPT.read_text().replace(before, after))
    monkeypatch.setattr(arena, "__file__", str(changed))
    old = dict(source_sha256={"scripts/stdio_arena.py": arena.sha256(old_source)})
    current = dict(source_sha256={"scripts/stdio_arena.py": arena.sha256(changed)})
    proof = dict(
        reviewed=True,
        reviewed_reason="Synthetic proof must fail when deadline changes",
        old_runner_source=str(old_source),
        old_runner_sha256=arena.sha256(old_source),
        new_runner_sha256=arena.sha256(changed),
    )
    with pytest.raises(ValueError, match="AST changed"):
        arena.verify_resume_metadata(old, current, proof)
    proof["reviewed"] = False
    with pytest.raises(ValueError, match="marked reviewed"):
        arena.verify_resume_metadata(old, current, proof)


def test_completed_runner_does_not_own_unrelated_shared_shell_process_group(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], cwd=tmp_path)
    try:
        record = arena.process_record(child.pid)
        record["pid"] = 987654321  # The prior runner is gone; another process shares its old shell group.
        arena.atomic_json(tmp_path / "runner.json", record)
        arena.assert_no_live_processes(tmp_path, [tmp_path / "agent/run.sh"] * 2)
    finally:
        child.kill()
        child.wait(timeout=2)

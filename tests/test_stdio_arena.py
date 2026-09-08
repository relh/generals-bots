"""Synthetic subprocess regressions for deadline ownership and fault accounting."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

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

"""Deadline-aware CPU matches between two reviewed local stdio run.sh agents.

Actual competition maps/rules, four seat/spawn-label assignments per board.
This local harness enforces response deadlines and50-fault forfeits, but does not
provide a network/filesystem sandbox or a hard memory cgroup. RSS is sampled.
"""

import argparse
import ast
import csv
import fcntl
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PASS = [1, 0, 0, 0, 0]


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class CaseAborted(RuntimeError):
    """The game evidence is already durable; persist its row before stopping."""

    def __init__(self, message, row):
        super().__init__(message)
        self.row = row


@dataclass(frozen=True)
class Limits:
    first_seconds: float = 10.0
    turn_seconds: float = 0.150
    fault_budget: int = 50
    max_stdout_bytes: int = 65536
    exit_seconds: float = 1.0


def parse_action(line, shape):
    try:
        action = [int(part) for part in line.split()]
    except ValueError:
        return PASS.copy(), "malformed"
    if len(action) != 5:
        return PASS.copy(), "malformed"
    if any(value < -(2**31) or value >= 2**31 for value in action):
        return PASS.copy(), "invalid_command"
    kind, row, col, direction, split = action
    if kind == 1:
        return PASS.copy(), None
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return PASS.copy(), "invalid_command"
    if kind == 2:
        return [2, row, col, 0, 0], None
    if kind != 0 or direction not in range(4) or split not in (0, 1):
        return PASS.copy(), "invalid_command"
    return action, None


class AgentProcess:
    """At most one outstanding request; late responses are discarded in its slot."""

    def __init__(self, command, cwd, environment, stderr_path, handshake=b"", limits=Limits()):
        self.limits = limits
        self.stderr = Path(stderr_path).open("wb")
        self.started = time.monotonic()
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            start_new_session=True,
            bufsize=0,
        )
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.outgoing = handshake
        self.buffer = b""
        self.pending = False
        self.closed = False
        self.dead_reason = None
        self.requests = 0
        self.faults = 0
        self.fault_reasons = Counter()
        self.forfeit_reason = None
        self.invalid_actions = 0
        self.peak_sampled_rss_bytes = 0
        self.peak_sampled_tree_rss_bytes = 0
        self.last_sample = 0.0
        self.result = None
        self.current_skipped = False
        self.stale_lines = 0

    def _sample_rss(self):
        now = time.monotonic()
        if now - self.last_sample < 0.02:
            return
        self.last_sample = now
        seen, pending, total = set(), [self.process.pid], 0
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                rss = next(
                    (int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0
                )
                total += rss
                if pid == self.process.pid:
                    self.peak_sampled_rss_bytes = max(self.peak_sampled_rss_bytes, rss)
                children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                pending.extend(map(int, children))
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.peak_sampled_tree_rss_bytes = max(self.peak_sampled_tree_rss_bytes, total)

    def _pump(self):
        self._sample_rss()
        if self.dead_reason:
            return
        if self.outgoing:
            try:
                sent = os.write(self.process.stdin.fileno(), self.outgoing)
                self.outgoing = self.outgoing[sent:]
            except BlockingIOError:
                pass
            except BrokenPipeError:
                self.dead_reason = "process_exit"
        while not self.dead_reason:
            try:
                chunk = os.read(self.process.stdout.fileno(), 4096)
            except BlockingIOError:
                break
            if not chunk:
                self.closed = True
                break
            self.buffer += chunk
            if len(self.buffer) > self.limits.max_stdout_bytes:
                self.dead_reason = "stdout_overflow"
                self._kill_group(signal.SIGKILL)
                break

    def _discard_buffered_lines(self):
        while b"\n" in self.buffer:
            _, self.buffer = self.buffer.split(b"\n", 1)
            self.stale_lines += 1
            self.pending = False

    def begin(self, payload, shape):
        self.shape = shape
        self.result = None
        self.frame_started = self.started if self.requests == 0 else time.monotonic()
        self.deadline = self.frame_started + (
            self.limits.first_seconds if self.requests == 0 else self.limits.turn_seconds
        )
        self.requests += 1
        # Only lines received BEFORE sending this request can be discarded as
        # old/unsolicited. If an older request is still pending, skip this frame.
        self._pump()
        self._discard_buffered_lines()
        self.current_skipped = self.pending
        if self.dead_reason or self.closed or self.process.poll() is not None:
            self.current_skipped = False
            reason = "process_exit" if self.process.poll() is not None else self.dead_reason or "missing_reply"
            return self._finish(PASS.copy(), reason)
        if not self.current_skipped and not self.dead_reason and not self.closed:
            # Partial unsolicited output cannot safely be assigned a frame.
            if self.buffer:
                self.pending = True
                self.current_skipped = True
            else:
                self.outgoing += payload
                self.pending = True
        self._pump()

    def _finish(self, action, reason):
        if reason in ("process_exit", "stdout_overflow"):
            self.forfeit_reason = reason
        elif reason:
            self.faults += 1
            self.fault_reasons[reason] += 1
        self.result = dict(
            action=action,
            fault=reason,
            response_seconds=time.monotonic() - self.frame_started,
            skipped_observation=self.current_skipped,
            stale_lines=self.stale_lines,
        )
        return self.result

    def poll(self):
        # Exiting forfeits even if the process wrote a valid line before exiting,
        # or already supplied this turn's response while its opponent is pending.
        if self.process.poll() is not None:
            return self._finish(PASS.copy(), "process_exit")
        if self.result is not None:
            return self.result
        self._pump()
        now = time.monotonic()
        if self.current_skipped:
            self._discard_buffered_lines()
            if not self.pending or self.dead_reason or self.closed or now >= self.deadline:
                return self._finish(PASS.copy(), "pending_late_reply")
            return None
        if b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            self.pending = False
            if now > self.deadline:
                self.stale_lines += 1
                return self._finish(PASS.copy(), "deadline")
            action, reason = parse_action(line, self.shape)
            if reason == "invalid_command":
                self.invalid_actions += 1
                result = self._finish(PASS.copy(), None)
                result["invalid_action"] = True
                result["reply"] = line[:1024].decode(errors="replace")
                return result
            result = self._finish(action, reason)
            result["reply"] = line[:1024].decode(errors="replace")
            return result
        if self.dead_reason or self.closed or self.process.poll() is not None:
            reason = "process_exit" if self.process.poll() is not None else self.dead_reason or "missing_reply"
            return self._finish(PASS.copy(), reason)
        if now >= self.deadline:
            return self._finish(PASS.copy(), "deadline")
        return None

    def replace_invalid_action(self):
        self.invalid_actions += 1
        self.result.update(original_action=self.result["action"], action=PASS.copy(), invalid_action=True)

    def _kill_group(self, sig):
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass

    def close(self):
        report = dict(pid=self.process.pid, waits=[], errors=[], reaped=False)
        try:
            try:
                self.process.stdin.close()
            except OSError as error:
                report["errors"].append(f"stdin_close: {error}")
            for stage, sig in (("eof", None), ("term", signal.SIGTERM), ("kill", signal.SIGKILL)):
                try:
                    if sig is not None:
                        self._kill_group(sig)
                    self.process.wait(timeout=self.limits.exit_seconds)
                    report["waits"].append(dict(stage=stage, timed_out=False))
                    report["reaped"] = True
                    break
                except subprocess.TimeoutExpired:
                    report["waits"].append(dict(stage=stage, timed_out=True))
                except OSError as error:
                    report["errors"].append(f"{stage}: {error}")
            # One final non-blocking reap can observe a just-completed SIGKILL.
            report["reaped"] = self.process.poll() is not None
            report["returncode"] = self.process.returncode
            report["final_reap_timeout"] = not report["reaped"]
        finally:
            # Clean descendants even if the shell exited or one wait failed.
            try:
                self._kill_group(signal.SIGKILL)
            except OSError as error:
                report["errors"].append(f"final_group_kill: {error}")
            for name, stream in (("stdout", self.process.stdout), ("stderr", self.stderr)):
                try:
                    stream.close()
                except OSError as error:
                    report["errors"].append(f"{name}_close: {error}")
        return report


def close_all(agents):
    reports = []
    for agent in agents:
        try:
            reports.append(agent.close())
        except BaseException as error:  # noqa: BLE001 - attempt every child's cleanup even on interruption
            reports.append(dict(pid=agent.process.pid, reaped=False, errors=[repr(error)]))
    return reports


def ask_pair(agents, payloads, shape):
    for agent, payload in zip(agents, payloads):
        agent.begin(payload, shape)
    while any(agent.result is None for agent in agents):
        for agent in agents:
            agent.poll()
        if any(agent.forfeit_reason or agent.faults >= agent.limits.fault_budget for agent in agents):
            for agent in agents:
                if agent.result is None:
                    agent.result = dict(
                        action=PASS.copy(),
                        fault=None,
                        response_seconds=None,
                        skipped_observation=False,
                        stale_lines=agent.stale_lines,
                        match_ended_before_reply=True,
                    )
            break
        unresolved = [agent for agent in agents if agent.result is None]
        if unresolved:
            timeout = min(0.005, max(0, min(agent.deadline for agent in unresolved) - time.monotonic()))
            readers = [agent.process.stdout for agent in unresolved]
            writers = [agent.process.stdin for agent in unresolved if agent.outgoing]
            select.select(readers, writers, [], timeout)
    return [agent.result for agent in agents]


def identity(run_sh):
    hashes = {}
    for path in sorted(run_sh.parent.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and not path.is_symlink():
            hashes[str(path.relative_to(run_sh.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    result = dict(run_sh=str(run_sh), directory_sha256=digest, file_sha256=hashes)
    for name, path in (
        ("manifest", run_sh.parent / "manifest.json"),
        ("provenance", run_sh.parent.parent / "provenance.json"),
    ):
        if path.is_file():
            result[name] = json.loads(path.read_text())
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_record(pid):
    """PID plus Linux start ticks prevents confusing a reused PID with a child."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=int(pid), state=fields[0], group=int(fields[2]), start_ticks=fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


class RunLock:
    def __init__(self, output):
        self.stream = (output / ".runner.lock").open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            raise ValueError("Another runner holds this output directory") from error

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def record_children(agents, output):
    atomic_json(
        output / "children.json",
        [
            dict(process_record(agent.process.pid) or {}, pid=agent.process.pid, group=agent.process.pid)
            for agent in agents
        ],
    )


def assert_no_live_processes(output, paths, *, legacy=False):
    recorded = []
    runner = output / "runner.json"
    if runner.is_file():
        recorded.append(dict(json.loads(runner.read_text()), _runner_record=True))
    children = output / "children.json"
    if children.is_file():
        recorded.extend(json.loads(children.read_text()))
    for evidence in output.glob("game-*.json"):
        recorded.extend(json.loads(evidence.read_text()).get("processes", []))
    for old in recorded:
        live = process_record(old["pid"])
        if (
            live
            and live["pid"] != os.getpid()
            and (not old.get("start_ticks") or old["start_ticks"] == live["start_ticks"])
        ):
            raise ValueError(f"Prior runner/child is still present: PID {live['pid']}")
    # Also detect legacy children whose runner died before writing a trace.
    roots = {path.parent.resolve() for path in paths} if legacy else set()
    groups = {old["group"] for old in recorded if old.get("group") and not old.get("_runner_record")}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = (entry / "cwd").resolve(strict=True)
            arguments = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
            same_output = False
            if "--output" in arguments and any("stdio_arena.py" in argument for argument in arguments):
                target = Path(arguments[arguments.index("--output") + 1])
                same_output = (cwd / target).resolve() == output.resolve()
            proc = process_record(int(entry.name))
            if cwd in roots or same_output or (proc and proc["group"] in groups):
                raise ValueError(f"Live runner/agent process prevents resume: PID {entry.name}")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def gameplay_ast(source):
    tree = ast.parse(source)
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    selected = {
        name: ast.dump(definitions[name], include_attributes=False)
        for name in ("Limits", "parse_action", "ask_pair", "physically_valid")
    }
    selected["PASS"] = [
        ast.dump(node, include_attributes=False)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PASS" for target in node.targets)
    ]
    for method in definitions["AgentProcess"].body:
        if isinstance(method, ast.FunctionDef) and method.name != "close":
            selected[f"AgentProcess.{method.name}"] = ast.dump(method, include_attributes=False)
    body = definitions["play_case"].body
    game_try = next(node for node in body if isinstance(node, ast.Try))

    class RemoveOwnershipJournal(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "record_children"
                and ast.unparse(node.value) == "record_children(processes, output)"
            ):
                return None
            return self.generic_visit(node)

    game_try = RemoveOwnershipJournal().visit(game_try)
    selected["play_case.gameplay"] = [ast.dump(node, include_attributes=False) for node in game_try.body]
    # Imports, observation preparation and compile-before-spawn timing are protected too.
    setup = body[: body.index(game_try)]
    setup = [node for node in setup if not (isinstance(node, ast.Assign) and ast.unparse(node) == "play_error = None")]
    selected["play_case.setup"] = [ast.dump(node, include_attributes=False) for node in setup]
    return selected


def verify_resume_metadata(old, current, equivalence=None):
    if equivalence and equivalence.get("reviewed") is not True:
        raise ValueError("Source equivalence must explicitly be marked reviewed")
    fields = (
        "candidate",
        "opponent",
        "python",
        "runtime",
        "cpus",
        "engine_cpu",
        "seed",
        "boards",
        "diagnostic_turns",
        "limits",
        "parent_jax_version",
        "rules",
        "agent_rng",
        "late_reply_policy",
    )
    for field in fields:
        if old.get(field) != current.get(field):
            raise ValueError(f"Resume identity/options mismatch: {field}")
    before, after = old["source_sha256"], current["source_sha256"]
    critical = {
        name
        for name in set(before) | set(after)
        if name.startswith(("generals/core/", "generals/modifiers/"))
        or name in ("generals/evaluation/arena.py", "generals/evaluation/scenarios.py", "competition/protocol.py")
    }
    for name in critical:
        if before.get(name) == after.get(name):
            continue
        attested = (equivalence or {}).get("legacy_source_attestations", {}).get(name)
        if name not in before and attested == after.get(name) and equivalence.get("reviewed_reason", "").strip():
            continue
        raise ValueError(f"Resume gameplay source mismatch or missing identity: {name}")
    runner = "scripts/stdio_arena.py"
    proof = None
    if before[runner] != after[runner]:
        if not equivalence or not equivalence.get("reviewed_reason", "").strip():
            raise ValueError("Changed runner requires explicit reviewed source equivalence")
        if (equivalence.get("old_runner_sha256"), equivalence.get("new_runner_sha256")) != (
            before[runner],
            after[runner],
        ):
            raise ValueError("Source equivalence does not name these exact runner hashes")
        old_source = Path(equivalence["old_runner_source"])
        if sha256(old_source) != before[runner]:
            raise ValueError("Reviewed old runner snapshot hash mismatch")
        protected = gameplay_ast(old_source.read_text())
        new_protected = gameplay_ast(Path(__file__).read_text())
        if protected != new_protected:
            raise ValueError("Gameplay/action/deadline AST changed; cleanup equivalence is invalid")
        proof = dict(
            equivalence,
            verified_unchanged_ast=sorted(protected),
            allowed_bookkeeping_call="record_children(processes, output) after each spawn",
        )
    elif equivalence:
        proof = dict(equivalence)
    if old.get("boards_sha256") and old["boards_sha256"] != current["boards_sha256"]:
        raise ValueError("Stored board archive changed")
    return proof


def validate_completed_rows(output, cases):
    """Reject partial/duplicate/conflicting data; recover durable complete JSON rows."""
    expected = {case["game_id"]: case for case in cases}
    rows, seen, recovered = [], set(), []
    csv_path = output / "games.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            raw = list(reader)
        if not reader.fieldnames or (not raw and csv_path.stat().st_size == 0):
            raise ValueError("Partial or empty existing CSV")
    else:
        raw = []
    for row in raw:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Partial CSV row")
        game_id = int(row["game_id"])
        if game_id in seen:
            raise ValueError("Duplicate completed game ID")
        evidence = output / f"game-{game_id:04d}.json"
        if not evidence.is_file():
            raise ValueError("Completed CSV row has no game evidence")
        details = json.loads(evidence.read_text())
        recorded = details["game"]
        if {k: "" if v is None else str(v) for k, v in recorded.items()} != row:
            raise ValueError("CSV row conflicts with game evidence")
        rows.append(recorded)
        seen.add(game_id)
    for evidence in sorted(output.glob("game-*.json")):
        details = json.loads(evidence.read_text())
        row = details["game"]
        if evidence.name != f"game-{int(row['game_id']):04d}.json":
            raise ValueError("Game evidence filename/ID conflict")
        if int(row["game_id"]) not in seen:
            rows.append(row)
            recovered.append(int(row["game_id"]))
            seen.add(int(row["game_id"]))
    for row in rows:
        game_id = int(row["game_id"])
        if game_id not in expected or row["result"] not in ("win", "loss", "draw"):
            raise ValueError("Partial/unfinished or unexpected case cannot be resumed")
        if any(str(row.get(name)) != str(value) for name, value in expected[game_id].items() if name != "grid"):
            raise ValueError("Completed row does not match its stored-board case")
        if "grid" in expected[game_id] and (int(row["height"]), int(row["width"])) != expected[game_id]["grid"].shape:
            raise ValueError("Completed row has the wrong board shape")
        if row["reason"] not in ("terminal", "turn_cap_draw", "fault_forfeit", "process_forfeit", "both_forfeit"):
            raise ValueError("Unknown or incomplete termination reason")
        details = json.loads((output / f"game-{game_id:04d}.json").read_text())
        if details.get("cleanup", {}).get("status") == "pending":
            raise ValueError("Partial game cleanup evidence cannot be resumed")
        if not details.get("frames"):
            raise ValueError("Completed case has no frame evidence")
        totals = [0, 0]
        reasons = [Counter(), Counter()]
        for frame in details["frames"]:
            replies = frame.get("replies")
            if not isinstance(replies, list) or len(replies) != 2:
                raise ValueError("Partial frame reply evidence")
            for player, reply in enumerate(replies):
                action = reply.get("action")
                if not isinstance(action, list) or len(action) != 5 or any(type(value) is not int for value in action):
                    raise ValueError("Invalid applied-action evidence")
                parsed, invalid = parse_action(
                    " ".join(map(str, action)).encode(), (int(row["height"]), int(row["width"]))
                )
                if invalid or parsed != action:
                    raise ValueError("Invalid applied-action evidence")
                if "fault" not in reply or "response_seconds" not in reply:
                    raise ValueError("Partial reply timing/fault evidence")
                fault = reply["fault"]
                if fault not in (
                    None,
                    "process_exit",
                    "stdout_overflow",
                    "deadline",
                    "missing_reply",
                    "pending_late_reply",
                    "malformed",
                ):
                    raise ValueError("Unknown reply fault")
                seconds = reply["response_seconds"]
                if seconds is None:
                    if not reply.get("match_ended_before_reply"):
                        raise ValueError("Unexplained missing response timing")
                elif not isinstance(seconds, (int, float)) or not 0 <= seconds < float("inf"):
                    raise ValueError("Invalid response timing")
                if fault and fault not in ("process_exit", "stdout_overflow"):
                    totals[player] += 1
                    reasons[player][fault] += 1
            if frame.get("faults") != totals:
                raise ValueError("Missing or conflicting cumulative frame faults")
        processes = details.get("processes")
        if not isinstance(processes, list) or len(processes) != 2:
            raise ValueError("Missing process evidence")
        for player, process in enumerate(processes):
            if (
                type(process.get("pid")) is not int
                or process.get("faults") != totals[player]
                or process.get("fault_reasons") != dict(reasons[player])
            ):
                raise ValueError("Conflicting process fault evidence")
        seat = int(row["seat"])
        if row.get("candidate_faults") != totals[seat] or row.get("opponent_faults") != totals[1 - seat]:
            raise ValueError("CSV fault totals conflict with complete frame trace")
        if [frame["turn"] for frame in details["frames"]] != list(range(len(details["frames"]))):
            raise ValueError("Partial or duplicate frame sequence")
        expected_frames = int(row["turns"]) + int(row["reason"] in ("fault_forfeit", "process_forfeit", "both_forfeit"))
        if len(details["frames"]) != expected_frames:
            raise ValueError("Partial game frame evidence")
        winner = int(row["winner"])
        outcome = "draw" if winner < 0 else "win" if winner == int(row["seat"]) else "loss"
        if row["result"] != outcome:
            raise ValueError("Conflicting winner/result")
    return sorted(rows, key=lambda row: int(row["game_id"])), recovered


def physically_valid(state, player, action):
    from generals.modifiers.build_castles import build_cost_grid

    kind, row, col, direction, _ = action
    if kind == 1:
        return True
    if not bool(state.ownership[player, row, col]):
        return False
    if kind == 2:
        return not bool(state.generals[row, col] | state.castles[row, col]) and int(state.armies[row, col]) >= int(
            build_cost_grid(state, player)[row, col]
        )
    dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][direction]
    dest_r, dest_c = row + dr, col + dc
    h, w = state.armies.shape
    return (
        int(state.armies[row, col]) > 1 and 0 <= dest_r < h and 0 <= dest_c < w and bool(state.passable[dest_r, dest_c])
    )


def play_case(paths, grid, case, rules, output, environment, cpus, diagnostic_turns=None):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from competition.protocol import encode_handshake, encode_observation
    from generals.core import game
    from generals.evaluation.arena import transition

    state = game.create_initial_state(jnp.asarray(grid))
    shape = state.armies.shape
    observe = jax.jit(lambda s: (game.get_observation(s, 0), game.get_observation(s, 1)))
    # Engine compilation is outside the agents' first-response budget.
    jax.block_until_ready(transition(state, jnp.array([PASS, PASS]), rules))
    observations = observe(state)
    payloads = [encode_observation(obs).encode() for obs in observations]
    processes, frames, reason, winner = [], [], None, -1
    play_error = None
    start = time.monotonic()
    try:
        for player, path in enumerate(paths):
            command = ["taskset", "-c", str(cpus[player]), "bash", str(path)]
            processes.append(
                AgentProcess(
                    command,
                    path.parent,
                    environment,
                    output / f"game-{case['game_id']:04d}-player-{player}.stderr.log",
                    encode_handshake(player, *shape).encode(),
                )
            )
            record_children(processes, output)
        for turn in range(rules.max_turns):
            if turn:
                payloads = [encode_observation(obs).encode() for obs in observe(state)]
            replies = ask_pair(processes, payloads, shape)
            for player, (agent, reply) in enumerate(zip(processes, replies)):
                if not reply["fault"] and not physically_valid(state, player, reply["action"]):
                    agent.replace_invalid_action()
            frames.append(dict(turn=turn, replies=replies, faults=[agent.faults for agent in processes]))
            forfeits = [bool(agent.forfeit_reason) or agent.faults >= agent.limits.fault_budget for agent in processes]
            if any(forfeits):
                winner = -1 if all(forfeits) else 1 - forfeits.index(True)
                reason = (
                    "both_forfeit"
                    if all(forfeits)
                    else ("process_forfeit" if processes[forfeits.index(True)].forfeit_reason else "fault_forfeit")
                )
                break
            state, info = transition(state, jnp.array([reply["action"] for reply in replies]), rules)
            if bool(info.is_done):
                winner, reason = int(info.winner), "terminal"
                break
            if diagnostic_turns and turn + 1 >= diagnostic_turns:
                reason = "diagnostic_cutoff"
                break
        if reason is None:
            reason = "turn_cap_draw"
    except BaseException as error:  # noqa: BLE001 - persist partial evidence and clean children before abort
        play_error = error
        reason = "runner_error"
    evidence = output / f"game-{case['game_id']:04d}.json"
    row = None
    details = dict(case=case, frames=frames, observed_winner=winner, observed_reason=reason)
    try:
        seat = case["seat"]
        result = (
            "unfinished"
            if reason in ("diagnostic_cutoff", "runner_error")
            else "draw"
            if winner < 0
            else "win"
            if winner == seat
            else "loss"
        )

        def agent_value(player, name):
            return getattr(processes[player], name) if player < len(processes) else 0

        row = {
            **case,
            "height": shape[0],
            "width": shape[1],
            "turns": int(state.time),
            "result": result,
            "winner": winner,
            "reason": reason,
            "wall_seconds": time.monotonic() - start,
            "candidate_faults": agent_value(seat, "faults"),
            "opponent_faults": agent_value(1 - seat, "faults"),
            "candidate_invalid_actions": agent_value(seat, "invalid_actions"),
            "opponent_invalid_actions": agent_value(1 - seat, "invalid_actions"),
            "candidate_peak_sampled_tree_rss": agent_value(seat, "peak_sampled_tree_rss_bytes"),
            "opponent_peak_sampled_tree_rss": agent_value(1 - seat, "peak_sampled_tree_rss_bytes"),
        }
        details = dict(
            game=row,
            frames=frames,
            processes=[
                dict(
                    pid=agent.process.pid,
                    start_ticks=(process_record(agent.process.pid) or {}).get("start_ticks"),
                    group=agent.process.pid,
                    returncode=agent.process.returncode,
                    faults=agent.faults,
                    fault_reasons=dict(agent.fault_reasons),
                    stale_lines=agent.stale_lines,
                    invalid_actions=agent.invalid_actions,
                    forfeit_reason=agent.forfeit_reason,
                    peak_sampled_rss_bytes=agent.peak_sampled_rss_bytes,
                    peak_sampled_tree_rss_bytes=agent.peak_sampled_tree_rss_bytes,
                )
                for agent in processes
            ],
        )
        details["cleanup"] = dict(status="pending")
        for label in ("candidate", "opponent"):
            for metric in (
                "first_response_seconds",
                "accepted_warm_median_seconds",
                "accepted_warm_p95_seconds",
                "accepted_warm_max_seconds",
            ):
                row[f"{label}_{metric}"] = None
        details["runner_error"] = repr(play_error) if play_error else None
        atomic_json(evidence, details)
        for label, player in (("candidate", seat), ("opponent", 1 - seat)):
            accepted = [
                frame["replies"][player]["response_seconds"]
                for frame in frames[1:]
                if frame["replies"][player]["fault"] is None
                and frame["replies"][player]["response_seconds"] is not None
            ]
            row[f"{label}_first_response_seconds"] = (
                frames[0]["replies"][player]["response_seconds"] if frames else None
            )
            row[f"{label}_accepted_warm_median_seconds"] = float(np.median(accepted)) if accepted else None
            row[f"{label}_accepted_warm_p95_seconds"] = float(np.quantile(accepted, 0.95)) if accepted else None
            row[f"{label}_accepted_warm_max_seconds"] = max(accepted) if accepted else None
    except BaseException as error:  # noqa: BLE001 - reporting failures must also clean every child
        play_error = play_error or error
        details["runner_error"] = repr(play_error)
    finally:
        cleanup = close_all(processes)
    failed = any(not report.get("reaped") or report.get("errors") for report in cleanup)
    details["cleanup"] = dict(status="failed" if failed else "complete", processes=cleanup)
    if row is not None:
        row["wall_seconds"] = time.monotonic() - start
    for record, agent in zip(details.get("processes", []), processes):
        record["returncode"] = agent.process.returncode
    atomic_json(evidence, details)
    if row is None:
        raise RuntimeError("Partial game evidence saved after reporting failure") from play_error
    if play_error or failed:
        raise CaseAborted("Game evidence saved; aborting after runner/cleanup failure", row) from play_error
    return row


def write_summary(output, rows):
    counts = Counter(row["result"] for row in rows)
    complete = sum(counts[name] for name in ("win", "loss", "draw"))
    atomic_json(
        output / "summary.json",
        dict(
            games=len(rows),
            **counts,
            completed_games=complete,
            win_rate=counts["win"] / complete if complete else None,
            candidate_faults=sum(row["candidate_faults"] for row in rows),
            opponent_faults=sum(row["opponent_faults"] for row in rows),
        ),
    )


def run_output(args, paths, environment, cpus, metadata):
    from dataclasses import asdict

    import numpy as np

    from generals.core.env import GeneralsEnv
    from generals.evaluation.arena import Rules
    from generals.evaluation.scenarios import make_suite, paired_cases

    output = args.output
    original_path = output / "metadata.json"
    board_path = output / "boards.npz"
    if not args.resume and any((output / name).exists() for name in ("metadata.json", "boards.npz", "games.csv")):
        raise ValueError("Output already contains a run; use --resume without overwriting evidence")
    legacy = args.resume and not json.loads(original_path.read_text()).get("segments_required")
    assert_no_live_processes(output, paths, legacy=legacy)
    proof, board_verification = None, "original_archive_hash"
    if args.resume:
        old = json.loads(original_path.read_text())
        with np.load(board_path, allow_pickle=False) as saved:
            if set(saved.files) != {str(i) for i in range(args.boards)}:
                raise ValueError("Stored board IDs/count changed")
            boards = [saved[str(i)].copy() for i in range(args.boards)]
        env = GeneralsEnv(mode="competition")
        rules = Rules(env.truncation, env.build_castles, env.deathtouch_turn)
        metadata.update(rules=asdict(rules), boards_sha256=sha256(board_path))
        equivalence = json.loads(args.resume_equivalence.read_text()) if args.resume_equivalence else None
        proof = verify_resume_metadata(old, metadata, equivalence)
        if "boards_sha256" not in old:
            regenerated, regenerated_rules = make_suite("competition", args.seed, args.boards)
            if regenerated_rules != rules or any(
                a.dtype != b.dtype or not np.array_equal(a, b) for a, b in zip(boards, regenerated)
            ):
                raise ValueError("Legacy stored boards differ from verified seeded generation")
            board_verification = "legacy_archive_regenerated_from_verified_source_and_seed_exact_arrays"
    else:
        boards, rules = make_suite("competition", args.seed, args.boards)
        metadata["rules"] = asdict(rules)
    cases = []
    for game_id, case in enumerate(paired_cases(boards, args.seed + 1)):
        case.pop("action_seed")
        cases.append(dict(case, game_id=game_id))
    rows, recovered = validate_completed_rows(output, cases) if args.resume else ([], [])
    completed = {int(row["game_id"]) for row in rows}
    pending = [case for case in cases if case["game_id"] not in completed]
    if args.check_resume:
        print(
            json.dumps(
                dict(
                    resume_valid=True,
                    completed_game_ids=sorted(completed),
                    pending_game_ids=[case["game_id"] for case in pending],
                    recovered_game_ids=recovered,
                    source_equivalence=proof,
                    board_verification=board_verification,
                )
            ),
            flush=True,
        )
        return
    if not args.resume:
        np.savez_compressed(board_path, **{str(i): board for i, board in enumerate(boards)})
        metadata["boards_sha256"] = sha256(board_path)
        atomic_json(original_path, metadata)
    segments = output / "segments"
    segments.mkdir(exist_ok=True)
    indexes = [int(path.stem) for path in segments.glob("*.json") if path.stem.isdigit()]
    index = max(indexes, default=-1) + 1
    segment = dict(
        metadata,
        segment_index=index,
        original_metadata_sha256=sha256(original_path),
        resumed=args.resume,
        completed_game_ids=sorted(completed),
        recovered_game_ids=recovered,
        pending_game_ids=[case["game_id"] for case in pending],
        source_equivalence=proof,
        board_verification=board_verification,
        runner=process_record(os.getpid()),
    )
    atomic_json(segments / f"{index:04d}.json", segment)
    shutil.copyfile(__file__, segments / f"{index:04d}-runner.py")
    atomic_json(output / "runner.json", process_record(os.getpid()))
    csv_path = output / "games.csv"
    if recovered:
        if csv_path.exists():
            shutil.copyfile(csv_path, segments / f"{index:04d}-original-games.csv")
        temporary = csv_path.with_suffix(".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(csv_path)
    fieldnames = None
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open(newline="") as stream:
            fieldnames = next(csv.reader(stream))
    if rows:
        write_summary(output, rows)
    status, failure = "complete", None
    try:
        with csv_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames) if fieldnames else None
            for original_case in pending:
                case = dict(original_case)
                grid = case.pop("grid")
                # Old interrupted runs can have stderr without a completed trace.
                # Preserve that evidence before rerunning only the missing case.
                leftovers = list(output.glob(f"game-{case['game_id']:04d}*"))
                if leftovers:
                    archive = segments / f"{index:04d}-prior-partials"
                    archive.mkdir(exist_ok=True)
                    for path in leftovers:
                        path.rename(archive / path.name)
                seats = paths if case["seat"] == 0 else paths[::-1]
                error = None
                try:
                    row = play_case(seats, grid, case, rules, output, environment, cpus, args.diagnostic_turns)
                except CaseAborted as stopped:
                    row, error = stopped.row, stopped
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
                write_summary(output, rows)
                print(json.dumps(row), flush=True)
                if error:
                    raise error
    except BaseException as error:
        status, failure = "aborted", repr(error)
        raise
    finally:
        atomic_json(
            segments / f"{index:04d}.result.json",
            dict(
                status=status,
                error=failure,
                completed_game_ids=[int(row["game_id"]) for row in rows if row["result"] in ("win", "loss", "draw")],
            ),
        )


def main():
    # competition/ is repo tooling, not an installed generals package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("opponent", type=Path)
    parser.add_argument("--boards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--candidate-bundle", type=Path, help="original archive retained only for its identity")
    parser.add_argument("--opponent-bundle", type=Path, help="original archive retained only for its identity")
    parser.add_argument("--cpus", type=int, nargs=2)
    parser.add_argument("--engine-cpu", type=int, help="separate engine core; defaults to an available non-bot core")
    parser.add_argument("--diagnostic-turns", type=int, help="stop early and classify games as unfinished")
    parser.add_argument("--resume", action="store_true", help="Strictly validate and continue only missing cases")
    parser.add_argument(
        "--resume-equivalence", type=Path, help="Explicit reviewed cleanup-only runner hash equivalence"
    )
    parser.add_argument(
        "--check-resume", action="store_true", help="Validate resume and print its plan without launching agents"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.resume_equivalence or args.check_resume) and not args.resume:
        parser.error("resume-equivalence/check-resume require --resume")
    if args.boards < 1 or (args.diagnostic_turns is not None and args.diagnostic_turns < 1):
        parser.error("counts must be positive")
    paths = [args.candidate.resolve(), args.opponent.resolve()]
    if not all(path.is_file() for path in paths):
        parser.error("both entrypoints must exist; this tool does not extract or build archives")
    available = sorted(os.sched_getaffinity(0))
    cpus = args.cpus or [available[-1], available[-2] if len(available) > 1 else available[-1]]
    if any(cpu not in available for cpu in cpus):
        parser.error("CPU assignment is outside allowed affinity")
    engine_cpus = [cpu for cpu in available if cpu not in cpus]
    engine_cpu = args.engine_cpu if args.engine_cpu is not None else (engine_cpus[0] if engine_cpus else available[0])
    if engine_cpu not in available:
        parser.error("engine CPU is outside allowed affinity")
    os.sched_setaffinity(0, {engine_cpu})
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_ENABLE_COMPILATION_CACHE",
            "XLA_FLAGS",
        )
    }
    # Preserve the venv bin directory: resolving its python symlink would select
    # the base interpreter and silently bypass the reviewed runtime packages.
    environment.update(
        PATH=str(args.python.absolute().parent) + os.pathsep + environment.get("PATH", ""),
        JAX_PLATFORMS="cpu",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    os.environ["JAX_PLATFORMS"] = "cpu"
    from dataclasses import asdict

    import jax

    args.output.mkdir(parents=True, exist_ok=True)
    versions = subprocess.run(
        [
            str(args.python.absolute()),
            "-c",
            "import sys,json,importlib.metadata as m;"
            "print(json.dumps({'python':sys.version,'executable':sys.executable,"
            "'packages':{p:m.version(p) for p in ['numpy','jax','jaxlib','scipy']}}))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    metadata = dict(
        candidate=identity(paths[0]),
        opponent=identity(paths[1]),
        python=str(args.python.absolute()),
        runtime=json.loads(versions.stdout),
        cpus=cpus,
        engine_cpu=engine_cpu,
        seed=args.seed,
        boards=args.boards,
        diagnostic_turns=args.diagnostic_turns,
        limits=asdict(Limits()),
        parent_jax_version=jax.__version__,
        memory="/proc sampled current RSS and process-tree RSS; no hard memory cgroup",
        isolation="CPU affinity and process groups; shared host, no filesystem/network sandbox",
        late_reply_policy="one outstanding frame; skip/fault subsequent frames until old reply is drained",
    )
    metadata["agent_rng"] = "Only maps are seeded by this harness; unmodified agents control their own PRNGs."
    metadata["constraints_source"] = "https://www.generals.bot/rules#match-constraints"
    for name, archive in (("candidate", args.candidate_bundle), ("opponent", args.opponent_bundle)):
        if archive:
            metadata[name]["archive"] = str(archive.resolve())
            metadata[name]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    metadata["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("core", "modifiers", "evaluation", "agents")
        for path in (root / "generals" / folder).glob("*.py")
    }
    metadata["source_sha256"]["scripts/stdio_arena.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata["source_sha256"]["competition/protocol.py"] = sha256(root / "competition/protocol.py")
    metadata["segments_required"] = True
    metadata["schema_version"] = 2
    with RunLock(args.output):
        run_output(args, paths, environment, cpus, metadata)


if __name__ == "__main__":
    main()

"""Deadline-aware CPU matches between two reviewed local stdio run.sh agents.

Actual competition maps/rules, four seat/spawn-label assignments per board.
This local harness enforces response deadlines and50-fault forfeits, but does not
provide a network/filesystem sandbox or a hard memory cgroup. RSS is sampled.
"""

import argparse
import csv
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PASS = [1, 0, 0, 0, 0]


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
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=self.limits.exit_seconds)
        except subprocess.TimeoutExpired:
            self._kill_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=self.limits.exit_seconds)
            except subprocess.TimeoutExpired:
                self._kill_group(signal.SIGKILL)
                self.process.wait(timeout=self.limits.exit_seconds)
        # Clean up descendants even when a shell launcher exited ahead of them.
        self._kill_group(signal.SIGKILL)
        self.process.stdout.close()
        self.stderr.close()


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
    finally:
        for agent in processes:
            agent.close()
    seat = case["seat"]
    result = (
        "unfinished" if reason == "diagnostic_cutoff" else "draw" if winner < 0 else "win" if winner == seat else "loss"
    )
    row = {
        **case,
        "height": shape[0],
        "width": shape[1],
        "turns": int(state.time),
        "result": result,
        "winner": winner,
        "reason": reason,
        "wall_seconds": time.monotonic() - start,
        "candidate_faults": processes[seat].faults,
        "opponent_faults": processes[1 - seat].faults,
        "candidate_invalid_actions": processes[seat].invalid_actions,
        "opponent_invalid_actions": processes[1 - seat].invalid_actions,
        "candidate_peak_sampled_tree_rss": processes[seat].peak_sampled_tree_rss_bytes,
        "opponent_peak_sampled_tree_rss": processes[1 - seat].peak_sampled_tree_rss_bytes,
    }
    for label, player in (("candidate", seat), ("opponent", 1 - seat)):
        accepted = [
            frame["replies"][player]["response_seconds"]
            for frame in frames[1:]
            if frame["replies"][player]["fault"] is None and frame["replies"][player]["response_seconds"] is not None
        ]
        row[f"{label}_first_response_seconds"] = frames[0]["replies"][player]["response_seconds"] if frames else None
        row[f"{label}_accepted_warm_median_seconds"] = float(np.median(accepted)) if accepted else None
        row[f"{label}_accepted_warm_p95_seconds"] = float(np.quantile(accepted, 0.95)) if accepted else None
        row[f"{label}_accepted_warm_max_seconds"] = max(accepted) if accepted else None
    details = dict(
        game=row,
        frames=frames,
        processes=[
            dict(
                pid=agent.process.pid,
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
    (output / f"game-{case['game_id']:04d}.json").write_text(json.dumps(details, indent=2) + "\n")
    return row


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    import numpy as np

    from generals.evaluation.scenarios import make_suite, paired_cases

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
    boards, rules = make_suite("competition", args.seed, args.boards)
    metadata["rules"] = asdict(rules)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(args.output / "boards.npz", **{str(i): board for i, board in enumerate(boards)})
    rows = []
    with (args.output / "games.csv").open("w", newline="") as stream:
        writer = None
        for game_id, case in enumerate(paired_cases(boards, args.seed + 1)):
            grid = case.pop("grid")
            case.pop("action_seed")  # The stdio protocol has no action-RNG seed field.
            case["game_id"] = game_id
            seats = paths if case["seat"] == 0 else paths[::-1]
            row = play_case(seats, grid, case, rules, args.output, environment, cpus, args.diagnostic_turns)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            counts = Counter(row["result"] for row in rows)
            complete = sum(counts[name] for name in ("win", "loss", "draw"))
            summary = dict(
                games=len(rows),
                **counts,
                completed_games=complete,
                win_rate=counts["win"] / complete if complete else None,
                candidate_faults=sum(row["candidate_faults"] for row in rows),
                opponent_faults=sum(row["opponent_faults"] for row in rows),
            )
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

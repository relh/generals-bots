"""Synchronous strategy evidence, never official deadline/deployment qualification.

run records every returned action without host-time fallback. parity compares the
unchanged Amin stdio entrypoint against the wire adapter on complete histories.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARCHIVE_SHA = "d69cc5d28e4805faf629c5c072a6d23011c727ca4a3b3da9f5e890ea82b78755"
PACKAGES = {"numpy": "2.4.6", "jax": "0.11.0", "jaxlib": "0.11.0", "scipy": "1.18.0"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if sys.version.split()[0] != "3.12.10" or versions != PACKAGES:
        raise ValueError("Run with the pinned official Python3.12.10/NumPy/JAX/SciPy runtime")
    return {"python": sys.version, "executable": sys.executable, "packages": versions}


def verify_external(directory, archive):
    directory, archive = Path(directory).resolve(), Path(archive).resolve()
    if sha(archive.read_bytes()) != ARCHIVE_SHA:
        raise ValueError("External archive is not the reviewed pinned Amin checkpoint")
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError("Duplicate archive entries")
        expected = {name: sha(bundle.read(name)) for name in bundle.namelist()}
    actual = {
        str(p.relative_to(directory)): sha(p.read_bytes())
        for p in directory.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    if expected != actual or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("External directory differs from the unmodified pinned archive")
    return {
        "directory": str(directory),
        "archive": str(archive),
        "archive_sha256": ARCHIVE_SHA,
        "file_sha256": expected,
        "directory_sha256": sha(json.dumps(actual, sort_keys=True).encode()),
    }


class Amin:
    """Original sources imported in a temporary namespace; no inference repair."""

    def __init__(self, directory, archive):
        self.identity = verify_external(directory, archive)
        directory = Path(self.identity["directory"])
        names = ("numpy_infer", "agent", "_strategy_amin_main")
        previous = {name: sys.modules.get(name) for name in names}
        old_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for name, filename in zip(names, ("numpy_infer.py", "agent.py", "main.py")):
                spec = importlib.util.spec_from_file_location(name, directory / filename)
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.main = module
        finally:
            sys.dont_write_bytecode = old_bytecode
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old

    def new_game(self, player, shape):
        return self.main.Agent(player_id=player, H=shape[0], W=shape[1])

    def act(self, agent, wire, shape):
        stream = io.StringIO(wire)
        obs = self.main._read_observation(stream, *shape, stream.readline())
        if stream.read():
            raise ValueError("Trailing observation bytes")
        return agent.act(obs)


def timed(call):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    result = call()
    # Conversion synchronizes JAX's asynchronous action before ending timing.
    action = [x.item() if hasattr(x, "item") else x for x in result]
    return action, {
        "wall_seconds": (time.perf_counter_ns() - wall) / 1e9,
        "process_cpu_seconds": (time.process_time_ns() - cpu) / 1e9,
    }


def applied_action(raw, state, player):
    from scripts.stdio_arena import PASS, parse_action, physically_valid

    action, fault = parse_action(" ".join(map(str, raw)).encode(), state.armies.shape)
    invalid = fault == "invalid_command"
    if fault is not None:
        action = PASS.copy()
    elif not physically_valid(state, player, action):
        invalid, action = True, PASS.copy()
    return action, {"invalid_action": invalid, "malformed_command": fault == "malformed"}


def candidate(name, rules):
    if name == "sentinel":
        from generals.agents.sentinel_agent import SentinelAgent

        cls = SentinelAgent
    elif name == "sentinel-v4":
        from generals.agents.sentinel_v4_agent import SentinelV4Agent

        cls = SentinelV4Agent
    elif name in ("sentinel-v5", "sentinel-v5-disabled"):
        from generals.agents.sentinel_v5_agent import SentinelV5Agent

        return SentinelV5Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            intercept_threats=name == "sentinel-v5",
        )
    elif name in ("sentinel-v6", "sentinel-v6-disabled"):
        from generals.agents.sentinel_v6_agent import SentinelV6Agent

        return SentinelV6Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            commit_defense=name == "sentinel-v6",
        )
    elif name in ("sentinel-v10", "sentinel-v10-disabled", "sentinel-v10-v6", "sentinel-v10-v6-disabled"):
        from generals.agents.sentinel_v10_agent import SentinelV10Agent

        return SentinelV10Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            parent_version=6 if "-v6" in name else 9,
            mobilize_home=not name.endswith("-disabled"),
        )
    elif name in ("sentinel-v9", "sentinel-v9-disabled"):
        from generals.agents.sentinel_v9_agent import SentinelV9Agent

        return SentinelV9Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            remember_enemy_general=name == "sentinel-v9",
        )
    elif name in (
        "sentinel-v8",
        "sentinel-v8-cheap",
        "sentinel-v8-direct",
        "sentinel-v8-disabled",
        "sentinel-v8-no-concentration",
    ):
        from generals.agents.sentinel_v8_agent import SentinelV8Agent

        return SentinelV8Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            concentrate_armies=name != "sentinel-v8-no-concentration",
            cheapest_collection=name in ("sentinel-v8", "sentinel-v8-cheap", "sentinel-v8-no-concentration"),
            direct_deployment=name in ("sentinel-v8", "sentinel-v8-direct", "sentinel-v8-no-concentration"),
        )
    elif name in ("sentinel-v7", "sentinel-v7-disabled"):
        from generals.agents.sentinel_v7_agent import SentinelV7Agent

        return SentinelV7Agent(
            build_castles=rules.build_castles,
            deathtouch_turn=rules.deathtouch_turn,
            max_turns=rules.max_turns,
            concentrate_armies=name == "sentinel-v7",
        )
    else:
        raise ValueError(f"Unknown strategy candidate: {name}")
    return cls(build_castles=rules.build_castles, deathtouch_turn=rules.deathtouch_turn, max_turns=rules.max_turns)


def engine():
    import jax

    from generals.core import game

    return jax.jit(lambda state: (game.get_observation(state, 0), game.get_observation(state, 1)))


def warm_engine(observe, grid, rules):
    import jax
    import jax.numpy as jnp

    from generals.core import game
    from generals.evaluation.arena import transition

    state = game.create_initial_state(jnp.asarray(grid))
    jax.block_until_ready(observe(state))
    jax.block_until_ready(transition(state, jnp.array([[1, 0, 0, 0, 0]] * 2), rules))


def play_case(case, rules, policy_name, amin, output, policy, observe):
    import jax
    import jax.numpy as jnp

    from competition.protocol import encode_observation
    from generals.core import game
    from generals.evaluation.arena import initial_memory, policy_step, transition

    case = dict(case)
    grid = case.pop("grid")
    # Match standalone Sentinel RNG initialization; Amin uses deterministic argmax.
    case.pop("action_seed", None)
    shape, seat = grid.shape, case["seat"]
    state = game.create_initial_state(jnp.asarray(grid))
    policy = policy if hasattr(policy, "step") else policy.act
    frames, winner, reason, error = [], -1, "turn_cap_draw", None
    try:
        memory = initial_memory(policy, shape)
        key = jax.random.PRNGKey(seat)
        enemy = amin.new_game(1 - seat, shape)
        for turn in range(rules.max_turns):
            observations = observe(state)
            wires = [encode_observation(obs) for obs in observations]
            key, action_key = jax.random.split(key)

            def own_call():
                nonlocal memory
                action, memory, _ = policy_step(policy, observations[seat], action_key, memory)
                return action

            frame = {
                "turn": turn,
                "players": [
                    {"role": "candidate" if player == seat else "opponent", "wire_observation": wire}
                    for player, wire in enumerate(wires)
                ],
            }
            frames.append(frame)
            # Inference calls are sequential; timing includes their own parsing/conversion.
            for player in range(2):
                call = own_call if player == seat else lambda: amin.act(enemy, wires[player], shape)
                raw, timing = timed(call)
                action, flags = applied_action(raw, state, player)
                frame["players"][player].update(raw_action=raw, applied_action=action, **timing, **flags)
            state, info = transition(state, jnp.asarray([p["applied_action"] for p in frame["players"]]), rules)
            if bool(info.is_done):
                winner, reason = int(info.winner), "terminal"
                break
    except BaseException as exc:  # noqa: BLE001 - save partial evidence, then propagate infrastructure failure
        reason, error = "infrastructure_error", repr(exc)
        raise
    finally:
        row = dict(
            case,
            suite="competition",
            opponent="amin-pinned",
            candidate=policy_name,
            height=shape[0],
            width=shape[1],
            turns=int(state.time),
            winner=winner,
            reason=reason,
            result="unfinished" if error else "draw" if winner < 0 else "win" if winner == seat else "loss",
        )
        row["score"] = None if error else {"win": 1.0, "draw": 0.5, "loss": 0.0}[row["result"]]
        save(output / f"game-{case['game_id']:04d}.json", {"game": row, "frames": frames, "error": error})
    return row


def parity_history(amin, history, timeout=120):
    """A bounded infrastructure timeout aborts parity; never substitutes an action."""
    shape, player = tuple(history["shape"]), history["player_id"]
    wires = history["observations"]
    if player not in (0, 1) or len(shape) != 2 or not wires:
        raise ValueError("Parity requires a player, two-dimensional shape and nonempty complete history")
    agent = amin.new_game(player, shape)
    expected = [list(map(int, amin.act(agent, wire, shape))) for wire in wires]
    directory = Path(amin.identity["directory"])
    environment = os.environ | {
        "PATH": str(Path(sys.executable).absolute().parent) + os.pathsep + os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child = subprocess.Popen(
        ["bash", str(directory / "run.sh")],
        cwd=directory,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = child.communicate(f"{player} {shape[0]} {shape[1]}\n" + "".join(wires), timeout=timeout)
    except BaseException:  # noqa: BLE001 - bounded group cleanup on timeout or interruption
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.communicate(timeout=5)
        raise
    if child.returncode != 0:
        raise RuntimeError(f"Pinned stdio exited {child.returncode}: {stderr}")
    actual = [list(map(int, line.split())) for line in stdout.splitlines()]
    if actual != expected:
        raise ValueError("Adapter/original-stdio action history mismatch")
    recorded = history.get("expected_original_raw_actions")
    if recorded is not None and actual != recorded:
        raise ValueError("Adapter/stdio actions disagree with recorded original actions")
    return {
        "shape": list(shape),
        "player_id": player,
        "frames": len(wires),
        "actions": actual,
        "history_sha256": sha(json.dumps(history, sort_keys=True).encode()),
        "stderr": stderr,
        "stdio_returncode": child.returncode,
        "exact_action_parity": True,
        "recorded_actions_checked": recorded is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "parity"])
    parser.add_argument("--external-directory", type=Path, required=True)
    parser.add_argument("--external-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=[
            "sentinel",
            "sentinel-v4",
            "sentinel-v5",
            "sentinel-v5-disabled",
            "sentinel-v6",
            "sentinel-v6-disabled",
            "sentinel-v7",
            "sentinel-v7-disabled",
            "sentinel-v8",
            "sentinel-v8-cheap",
            "sentinel-v8-direct",
            "sentinel-v8-disabled",
            "sentinel-v8-no-concentration",
            "sentinel-v9",
            "sentinel-v9-disabled",
            "sentinel-v10",
            "sentinel-v10-disabled",
            "sentinel-v10-v6",
            "sentinel-v10-v6-disabled",
        ],
        default="sentinel",
    )
    parser.add_argument("--boards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=83000)
    parser.add_argument(
        "--history",
        type=Path,
        action="append",
        default=[],
        help="JSON: shape, player_id, observations (complete ordered wire frames)",
    )
    parser.add_argument("--synthetic-shapes", action="store_true", help="Parity on all16 shapes,30frames each")
    args = parser.parse_args()
    if args.boards < 1:
        parser.error("boards must be positive")
    if args.output.exists():
        parser.error("refusing to overwrite output")
    os.environ["JAX_PLATFORMS"] = "cpu"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    args.output.mkdir(parents=True)
    status = {"state": "starting", "pid": os.getpid(), "mode": args.mode}
    save(args.output / "status.json", status)
    try:
        versions = runtime()
        from dataclasses import asdict

        import jax
        import numpy as np

        from generals.evaluation.scenarios import make_suite, paired_cases

        if any(device.platform != "cpu" for device in jax.devices()):
            raise ValueError("Strategy reference requires CPU")
        amin = Amin(args.external_directory, args.external_archive)
        files = [
            p for area in ("agents", "core", "modifiers", "evaluation") for p in (ROOT / "generals" / area).glob("*.py")
        ]
        files += [Path(__file__), ROOT / "scripts/stdio_arena.py", ROOT / "competition/protocol.py"]
        metadata = {
            "kind": "synchronous-strategy-only/1",
            "runtime": versions,
            "external": amin.identity,
            "candidate": args.candidate,
            "seed": args.seed,
            "boards": args.boards,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_hashes": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in files},
            "limits": "No response deadlines or host-time fallback. Strategy evidence only; "
            "not official deployment qualification. Inference wall/processCPU timings include first JIT.",
            "rng": "Sentinel PRNGKey(player), split once perturn; Amin original deterministic argmax.",
        }
        save(args.output / "metadata.json", metadata)
        if args.mode == "parity":
            histories = [json.loads(p.read_text()) for p in args.history]
            if args.synthetic_shapes:
                from scripts.probe_sentinel_bundle import frame

                histories.extend(
                    {"shape": [h, w], "player_id": 0, "observations": [frame(h, w, i, 42).decode() for i in range(30)]}
                    for h in range(18, 22)
                    for w in range(18, 22)
                )
            if not histories:
                raise ValueError("Supply --history or --synthetic-shapes")
            checks = []
            for index, history in enumerate(histories):
                save(args.output / f"history-{index:04d}.json", history)
                checks.append(parity_history(amin, history))
                save(args.output / "parity.json", {"checks": checks, "complete": False})
            save(args.output / "parity.json", {"checks": checks, "complete": True})
        else:
            boards, rules = make_suite("competition", args.seed, args.boards)
            metadata["rules"] = asdict(rules)
            save(args.output / "metadata.json", metadata)
            np.savez_compressed(args.output / "boards.npz", **{str(i): b for i, b in enumerate(boards)})
            metadata["boards_sha256"] = sha((args.output / "boards.npz").read_bytes())
            save(args.output / "metadata.json", metadata)
            policy = candidate(args.candidate, rules)
            observe, warmed = engine(), set()
            rows = []
            with (args.output / "games.csv").open("w", newline="") as stream:
                writer = None
                for index, case in enumerate(paired_cases(boards, args.seed + 1)):
                    if case["grid"].shape not in warmed:
                        warm_engine(observe, case["grid"], rules)
                        warmed.add(case["grid"].shape)
                    status.update(state="running", game_id=index, completed_games=len(rows))
                    save(args.output / "status.json", status)
                    with (args.output / f"game-{index:04d}.stderr.log").open("w") as stderr:
                        with contextlib.redirect_stderr(stderr):
                            row = play_case(
                                dict(case, game_id=index), rules, args.candidate, amin, args.output, policy, observe
                            )
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    status["completed_games"] = len(rows)
                    save(
                        args.output / "summary.json",
                        {
                            "games": len(rows),
                            **{result: sum(r["result"] == result for r in rows) for result in ("win", "loss", "draw")},
                        },
                    )
                    print(json.dumps(row), flush=True)
        verify_external(args.external_directory, args.external_archive)
        if any(sha((ROOT / name).read_bytes()) != value for name, value in metadata["source_hashes"].items()):
            raise RuntimeError("Source changed during reference run")
        status["state"] = "complete"
    except BaseException as exc:  # noqa: BLE001 - persist failure without replacing missing actions
        status.update(state="failed", error=repr(exc))
        raise
    finally:
        save(args.output / "status.json", status)


if __name__ == "__main__":
    main()

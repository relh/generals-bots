"""Cold single-core stdio deadline/RSS probe of an extracted Sentinel zip.

This is a sequential diagnostic, not an official sandbox or a strength match.
Late responses are faulted and drained without credit before the next frame.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import selectors
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


def frame(h, w, index, seed):
    """Consistent observation of a synthetic legal initial or midgame state."""
    rng = random.Random(seed + index)
    owners = [[0] * w for _ in range(h)]
    armies = [[0] * w for _ in range(h)]
    types = [[1] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if index and (c < max(2, w // 3) or c >= w - max(2, w // 3)):
                owners[r][c] = 1 if c < w // 2 else 2
                armies[r][c] = rng.randrange(1, 80)
            elif 1 < c < w - 2 and rng.random() < 0.22:
                types[r][c] = 2
    types[0][0], types[-1][-1] = 4, 4
    owners[0][0], owners[-1][-1] = 1, 2
    armies[0][0] = armies[-1][-1] = 1 if index == 0 else 40
    if index:
        types[h // 2][1] = types[h // 2][-2] = 3
    land = [sum(x == owner for row in owners for x in row) for owner in (1, 2)]
    total = [sum(armies[r][c] for r in range(h) for c in range(w) if owners[r][c] == owner) for owner in (1, 2)]
    visible = [
        [
            any(
                owners[rr][cc] == 1
                for rr in range(max(0, r - 1), min(h, r + 2))
                for cc in range(max(0, c - 1), min(w, c + 2))
            )
            for c in range(w)
        ]
        for r in range(h)
    ]
    for r in range(h):
        for c in range(w):
            if not visible[r][c]:
                types[r][c] = 5 if types[r][c] in (2, 3) else 0
                owners[r][c] = armies[r][c] = 0
    turn = 0 if index == 0 else 250 if index % 2 else 850
    lines = [f"{turn} {land[0]} {total[0]} {land[1]} {total[1]}"]
    lines.extend(" ".join(map(str, row)) for grid in (types, owners, armies) for row in grid)
    return ("\n".join(lines) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--shape", default="18x21")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument(
        "--build-cache", action="store_true", help="Run bundle build.sh and enable its persistent JIT cache"
    )
    cache.add_argument("--reuse-cache", action="store_true", help="Enable the cache already built in --work-dir")
    parser.add_argument(
        "--work-dir", type=Path, help="Retain extraction/cache here for subsequent fresh-process probes"
    )
    parser.add_argument("--build-timeout", type=float, default=300, help="Maximum build duration in seconds")
    args = parser.parse_args()
    h, w = map(int, args.shape.lower().split("x"))
    if min(h, w) < 4 or args.frames < 1:
        parser.error("dimensions >=4 and frames >=1 are required")
    if args.reuse_cache and not args.work_dir:
        parser.error("--reuse-cache requires --work-dir")
    affinity = sorted(os.sched_getaffinity(0))
    cpu = affinity[-1] if args.cpu is None else args.cpu
    if cpu not in affinity:
        parser.error("requested CPU is outside this process's affinity")
    report = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "cpu": cpu,
        "cache_mode": "built" if args.build_cache else "reused" if args.reuse_cache else "disabled",
        "shape": [h, w],
        "source": "https://www.generals.bot/rules#match-constraints",
        "limits": {"first_seconds": 10, "turn_seconds": 0.150, "fault_budget": 50, "rss_bytes": 2 * 1024**3},
        "qualification": "Single CPU affinity, externally shared host; timings may include contention. "
        "RSS high-water sampled from /proc, not an official cgroup. No network namespace is applied. "
        "Frames are synthetic, not an episode. Late replies are discarded after a bounded diagnostic drain; "
        "they are never assigned to the next observation.",
        "frames": [],
        "faults": 0,
        "peak_rss_bytes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    workspace = (
        contextlib.nullcontext(str(args.work_dir.resolve()))
        if args.work_dir
        else tempfile.TemporaryDirectory(prefix="sentinel-standalone-")
    )
    with workspace as temporary:
        work = Path(temporary)
        bundle = work / "bundle"
        bundle.mkdir(exist_ok=True)
        with zipfile.ZipFile(args.bundle) as archive:
            for info in archive.infolist():
                if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts:
                    raise ValueError("unsafe archive path")
            archive.extractall(bundle)
        report["manifest"] = json.loads((bundle / "manifest.json").read_text())
        env = os.environ.copy()
        for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS"):
            env.pop(name, None)
        env.update(
            PATH=str(args.python.absolute().parent) + os.pathsep + env.get("PATH", ""),
            JAX_PLATFORMS="cpu",
            PYTHONNOUSERSITE="1",
            JAX_ENABLE_COMPILATION_CACHE="true" if args.build_cache or args.reuse_cache else "false",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        # Both working directory and import path are outside the checkout. The
        # launch script must find its own package without the repo's installation.
        version_code = (
            "import sys,json,importlib.metadata as m;"
            "print(json.dumps({'python':sys.version.split()[0],"
            "'packages':{p:m.version(p) for p in ['jax','jaxlib','numpy','scipy']}}))"
        )
        versions = subprocess.run(
            [str(args.python.absolute()), "-c", version_code],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        report["runtime"] = json.loads(versions.stdout)
        expected = {"jax": "0.11.0", "jaxlib": "0.11.0", "numpy": "2.4.6", "scipy": "1.18.0"}
        report["version_mismatches"] = {
            p: {"expected": v, "actual": report["runtime"]["packages"][p]}
            for p, v in expected.items()
            if report["runtime"]["packages"][p] != v
        }
        if report["runtime"]["python"] != "3.12.10":
            report["version_mismatches"]["python"] = {"expected": "3.12.10", "actual": report["runtime"]["python"]}
        if args.build_cache:
            if not (bundle / "build.sh").is_file():
                parser.error("--build-cache requires a bundle containing build.sh")
            build_log = args.output.with_suffix(".build.log")
            build_started = time.perf_counter()
            with build_log.open("wb") as log:
                built = subprocess.run(
                    ["bash", str(bundle / "build.sh")],
                    cwd=bundle,
                    env=env,
                    stdout=log,
                    stderr=log,
                    timeout=args.build_timeout,
                    preexec_fn=lambda: os.sched_setaffinity(0, {cpu}),
                    check=False,
                )
            report["build"] = {
                "wall_seconds": time.perf_counter() - build_started,
                "exit_code": built.returncode,
                "log": str(build_log),
                "artifact_bytes": sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()),
                "artifact_files": sum(path.is_file() for path in bundle.rglob("*")),
            }
            if built.returncode:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                raise SystemExit(f"Build failed; see {build_log}")
        stderr_path = args.output.with_suffix(".stderr.log")
        with stderr_path.open("wb") as stderr:
            started = time.perf_counter()
            proc = subprocess.Popen(
                ["bash", str(bundle / "run.sh")],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=work,
                env=env,
                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}),
            )
            selector = selectors.DefaultSelector()
            selector.register(proc.stdout, selectors.EVENT_READ)
            buffer = b""

            def read_until(deadline):
                nonlocal buffer
                while time.perf_counter() < deadline:
                    try:
                        status = Path(f"/proc/{proc.pid}/status").read_text()
                        rss = max(
                            [
                                int(line.split()[1]) * 1024
                                for line in status.splitlines()
                                if line.startswith(("VmRSS:", "VmHWM:"))
                            ]
                            or [0]
                        )
                        report["peak_rss_bytes"] = max(report["peak_rss_bytes"], rss)
                        if rss > report["limits"]["rss_bytes"]:
                            proc.kill()
                            report["stop_reason"] = "sampled_memory_limit_exceeded"
                            return None
                    except FileNotFoundError:
                        pass
                    if b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        return line
                    if selector.select(min(0.005, max(0, deadline - time.perf_counter()))):
                        chunk = os.read(proc.stdout.fileno(), 4096)
                        if not chunk:
                            report["stop_reason"] = "process_exit"
                            return None
                        buffer += chunk
                return None

            try:
                proc.stdin.write(f"0 {h} {w}\n".encode())
                for index in range(args.frames):
                    # Preparation occurs before each measured observation delivery.
                    payload = frame(h, w, index, args.seed)
                    sent = started if index == 0 else time.perf_counter()
                    deadline = sent + (10 if index == 0 else 0.150)
                    proc.stdin.write(payload)
                    proc.stdin.flush()
                    line = read_until(deadline)
                    elapsed = time.perf_counter() - sent
                    fault = line is None or elapsed > (10 if index == 0 else 0.150)
                    row = {"index": index, "reply_seconds": elapsed, "deadline_seconds": 10 if index == 0 else 0.150}
                    if line is None and proc.poll() is None and "stop_reason" not in report:
                        late = read_until(time.perf_counter() + 5)
                        row["late_reply_drained"] = late is not None
                        row["late_reply_seconds"] = time.perf_counter() - sent
                        if late is None:
                            report["stop_reason"] = "late_reply_not_drained"
                    elif line is not None:
                        try:
                            action = [int(x) for x in line.split()]
                            if len(action) != 5:
                                raise ValueError
                            row["action"] = action
                        except ValueError:
                            fault = True
                            row["malformed_reply"] = line.decode(errors="replace")
                    row["fault"] = fault
                    report["faults"] += int(fault)
                    report["frames"].append(row)
                    if report["faults"] >= 50:
                        report["stop_reason"] = "fault_budget_exhausted"
                    if "stop_reason" in report:
                        break
            except BrokenPipeError:
                report["stop_reason"] = "process_exit"
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                report["exit_code"] = proc.returncode
                selector.close()
        report["stderr_log"] = str(stderr_path)
    times = [r["reply_seconds"] for r in report["frames"][1:] if not r["fault"]]
    report["warm_median_seconds"] = statistics.median(times) if times else None
    report["warm_max_seconds"] = max(times) if times else None
    report["all_deadlines_met"] = (
        report["faults"] == 0 and len(report["frames"]) == args.frames and report["exit_code"] == 0
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "all_deadlines_met",
                    "faults",
                    "peak_rss_bytes",
                    "version_mismatches",
                    "warm_median_seconds",
                    "warm_max_seconds",
                    "exit_code",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

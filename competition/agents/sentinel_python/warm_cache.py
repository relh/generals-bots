"""Populate the packaged CPU JIT cache offline for every competition rectangle."""

import argparse
import io
import json
import os
import time
from pathlib import Path

import jax
from main import make_agent, read_observation


def initial_frame(height, width):
    types = [[0] * width for _ in range(height)]
    owners = [[0] * width for _ in range(height)]
    armies = [[0] * width for _ in range(height)]
    for r in range(2):
        for c in range(2):
            types[r][c] = 1
    types[0][0], owners[0][0], armies[0][0] = 4, 1, 1
    lines = ["0 1 1 1 1"]
    lines.extend(" ".join(map(str, row)) for grid in (types, owners, armies) for row in grid)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=[f"{h}x{w}" for h in range(18, 22) for w in range(18, 22)])
    args = parser.parse_args()
    if os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "true").lower() == "false":
        raise RuntimeError("build requires JAX_ENABLE_COMPILATION_CACHE=true")
    started = time.perf_counter()
    agent = make_agent()
    key = jax.random.PRNGKey(0)
    report = {
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "shapes": [],
    }
    for shape in args.shapes:
        h, w = map(int, shape.split("x"))
        if not (18 <= h <= 21 and 18 <= w <= 21):
            raise ValueError("competition shapes must have each side in 18..21")
        before = time.perf_counter()
        obs = read_observation(io.StringIO(initial_frame(h, w)), h, w)
        key, action_key = jax.random.split(key)
        if hasattr(agent, "initial_memory"):
            action, _, _ = jax.block_until_ready(agent.step(obs, action_key, agent.initial_memory((h, w))))
        else:
            action = jax.block_until_ready(agent.act(obs, action_key))
        report["shapes"].append(
            {"shape": [h, w], "seconds": time.perf_counter() - before, "action": [int(x) for x in action]}
        )
        print(json.dumps(report["shapes"][-1]), flush=True)
    report["seconds"] = time.perf_counter() - started
    files = [p for p in Path(".").rglob("*") if p.is_file()]
    report["package_files_after_build"] = len(files)
    report["package_bytes_after_build"] = sum(p.stat().st_size for p in files)
    if report["package_files_after_build"] > 10000 or report["package_bytes_after_build"] > 512 * 1024**2:
        raise RuntimeError("built package exceeds the documented unpacked size/file limits")
    Path("build-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

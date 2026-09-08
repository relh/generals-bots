"""Bounded scalar/vmapped inference profiling on complete fog observations.

Preparation and host/device transfer are outside inference timing. Use an idle
machine for measurements; --smoke validates the tool and is not a benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def dimensions(value):
    try:
        h, w = (int(x) for x in value.lower().split("x"))
        if min(h, w) < 4:
            raise ValueError
        return h, w
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use HxW, each dimension at least 4") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--shapes", nargs="+", type=dimensions, default=[(8, 8), (12, 12), (18, 21)])
    parser.add_argument("--agents", nargs="+", choices=("sentinel", "expander", "hunter"),
                        default=["sentinel", "expander", "hunter"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="8x8 Sentinel, batch 2, one repeat; validation only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("batch-size/repeats must be positive; warmup must be nonnegative")
    if args.smoke:
        args.shapes, args.agents = [(8, 8)], ["sentinel"]
        args.batch_size, args.repeats, args.warmup = 2, 1, 0
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(ROOT))

    import jax
    import jax.numpy as jnp
    import numpy as np
    from generals.agents.expander_agent import ExpanderAgent
    from generals.agents.hunter_agent import HunterAgent
    from generals.agents.sentinel_agent import SentinelAgent
    from generals.core import game
    from generals.core.action import compute_valid_move_mask_obs

    sources = [Path(__file__), ROOT / "generals/agents/sentinel_agent.py",
               ROOT / "generals/agents/expander_agent.py", ROOT / "generals/agents/hunter_agent.py",
               ROOT / "generals/core/game.py", ROOT / "generals/core/observation.py",
               ROOT / "generals/core/action.py"]
    report = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "purpose": "smoke_validation_only" if args.smoke else "inference_benchmark",
        "contention": "Externally controlled; this tool does not establish machine isolation.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": [{"name": str(d), "kind": d.device_kind, "platform": d.platform} for d in jax.devices()],
        "host": {"platform": platform.platform(), "python": sys.version,
                 "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "versions": {"jax": jax.__version__, "numpy": np.__version__},
        "environment": {key: os.environ.get(key) for key in (
            "JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE")},
        "memory_semantics": "Device-wide allocator snapshots, possibly including earlier cases; not per-call peaks. "
                            "Null means this backend does not expose memory statistics.",
        "source": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                    capture_output=True, text=True, check=False).stdout.strip(),
                   "sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources}},
        "input_description": "Seeded legal synthetic midgame states: connected owned bands, visible and fog cells, "
                             "mountains outside owned bands, neutral castles in classic mode, owned castles in competition. "
                             "Observations come from game.get_observation with all fields populated. No gameplay strength claim.",
        "timing_semantics": "Tracing/lowering, compilation, first execution, and warmed synchronized calls measured separately. "
                            "Preparation, transfers, validation, and memory queries excluded from inference timing. "
                            "Median/p95 describe per-call latency; vmapped calls return batch-size actions.",
        "results": [],
    }

    def memory():
        stats = jax.devices()[0].memory_stats()
        return None if stats is None else {k: int(v) for k, v in stats.items()
                                           if isinstance(v, (int, np.integer))}

    def observations(h, w):
        # Construct actual GameStates before deriving observations so ownership,
        # neutral masks, fog, counts, and allied defaults are mutually consistent.
        rng = np.random.default_rng(args.seed + h * 1000 + w)
        grids, army_arrays, ownership_arrays = [], [], []
        band = max(2, w // 3)
        competition = h >= 18 and w >= 18
        for _ in range(args.batch_size):
            grid = np.zeros((h, w), np.int32)
            wall = rng.random((h, w)) < .16
            wall[:, :band] = wall[:, -band:] = False
            wall[h // 2, :] = False  # a guaranteed connection between both sides
            grid[wall] = -2
            if not competition:
                for r in (h // 3, 2 * h // 3):
                    grid[r, w // 2] = int(rng.integers(15, 40))
            grid[0, 0], grid[-1, -1] = 1, 2
            own = np.zeros((2, h, w), bool)
            own[0, :, :band], own[1, :, -band:] = True, True
            armies = np.where(own.any(0), rng.integers(1, 65, (h, w)), np.maximum(grid, 0))
            grids.append(grid)
            army_arrays.append(armies.astype(np.int32))
            ownership_arrays.append(own)
        state = jax.vmap(game.create_initial_state)(jnp.asarray(np.stack(grids)))
        own = jnp.asarray(np.stack(ownership_arrays))
        state = state._replace(armies=jnp.asarray(np.stack(army_arrays)), ownership=own,
                               ownership_neutral=state.passable & ~jnp.any(own, axis=1),
                               time=jnp.full((args.batch_size,), 850 if competition else 250, jnp.int32))
        if competition:
            castles = state.castles.at[:, h // 2, 1].set(True).at[:, h // 2, w - 2].set(True)
            state = state._replace(castles=castles)
        seats = jnp.arange(args.batch_size) % 2
        obs = jax.vmap(game.get_observation)(state, seats)
        assert all(x is not None for x in obs), "Observation must include allied fields"
        jax.block_until_ready(obs)
        return obs, competition

    for h, w in args.shapes:
        obs, competition = observations(h, w)
        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.batch_size)
        scalar_obs, scalar_key = jax.tree.map(lambda x: x[0], obs), keys[0]
        agents = {"sentinel": SentinelAgent(build_castles=competition,
                  deathtouch_turn=800 if competition else None, max_turns=1200 if competition else 800),
                  "expander": ExpanderAgent(), "hunter": HunterAgent()}
        for name in args.agents:
            outputs = {}
            for mode, arguments in (("scalar", (scalar_obs, scalar_key)), ("vmapped", (obs, keys))):
                fn = jax.jit(agents[name].act if mode == "scalar" else jax.vmap(agents[name].act))
                started = time.perf_counter()
                lowered = fn.lower(*arguments)
                lowering = time.perf_counter() - started
                started = time.perf_counter()
                compiled = lowered.compile()
                compilation = time.perf_counter() - started
                before_memory = memory()
                started = time.perf_counter()
                output = jax.block_until_ready(compiled(*arguments))
                first_execution = time.perf_counter() - started
                for _ in range(args.warmup):
                    jax.block_until_ready(compiled(*arguments))
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    output = jax.block_until_ready(compiled(*arguments))
                    times.append(time.perf_counter() - started)
                after_memory = memory()
                outputs[mode] = np.asarray(output)
                row = {"agent": name, "shape": [h, w], "mode": mode,
                       "rules": "competition" if competition else "classic",
                       "batch_size": 1 if mode == "scalar" else args.batch_size,
                       "lowering_seconds": lowering, "compile_seconds": compilation,
                       "first_execution_seconds": first_execution,
                       "warmed_seconds": times, "median_seconds": float(np.median(times)),
                       "p95_seconds": float(np.percentile(times, 95)),
                       "device_memory_before": before_memory, "device_memory_after": after_memory}
                row["actions_per_second"] = row["batch_size"] / row["median_seconds"]
                report["results"].append(row)
                print(json.dumps(row), flush=True)
            np.testing.assert_array_equal(outputs["scalar"], outputs["vmapped"][0])
            # Validate scalar physical move against the same complete observation.
            kind, r, c, direction, _ = outputs["scalar"]
            if kind == 0:
                assert bool(compute_valid_move_mask_obs(scalar_obs)[r, c, direction])
            else:
                assert kind in (1, 2) and (kind != 2 or competition)
    report["scalar_vmap_agreement"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Reproduce original 4x4 PPO stage timings and a semantics-equivalent scan.

Run from the repository root with the training extra installed. Device selection
happens before importing JAX. This tool does not modify a checkpoint or trainer.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def command_output(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Optional JAX profiler trace directory")
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive (warmup may be zero)")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(ROOT))

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from examples._experimental.ppo import train
    from generals.core import game
    from generals.core.action import compute_valid_move_mask

    report = {
        "command": sys.argv,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "cpu_model": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
            "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "packages": {
                name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "equinox", "optax", "numpy")
            },
            "environment": {
                k: os.environ.get(k)
                for k in (
                    "JAX_PLATFORMS",
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "LD_LIBRARY_PATH",
                )
            },
        },
        "source": {
            "commit": command_output(["git", "rev-parse", "HEAD"]),
            "status": command_output(["git", "status", "--short"]),
            "sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__).resolve(),
                    Path(train.__file__),
                    ROOT / "examples/_experimental/ppo/network.py",
                    ROOT / "generals/core/game.py",
                ]
            },
        },
        "timing_semantics": (
            "All timings synchronize all returned array leaves. First call includes tracing/compilation and execution; "
            "steady-state timings exclude first call and warmups. Stages overlap conceptually and must not be summed."
        ),
        "timings": {},
    }
    print(json.dumps(report["machine"]), flush=True)

    def measure(name, fn, *arguments, units=None):
        start = time.perf_counter()
        result = fn(*arguments)
        jax.block_until_ready(result)
        first = time.perf_counter() - start
        for _ in range(args.warmup):
            jax.block_until_ready(fn(*arguments))
        seconds = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = fn(*arguments)
            jax.block_until_ready(result)
            seconds.append(time.perf_counter() - start)
        median = statistics.median(seconds)
        values = {
            "first_call_seconds": first,
            "steady_seconds": seconds,
            "median_seconds": median,
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
        }
        if units:
            values["environment_steps_per_second"] = units / median
        report["timings"][name] = values
        print(f"{name}: {json.dumps(values)}", flush=True)
        return result

    key, net_key = jax.random.split(jax.random.PRNGKey(args.seed))
    network = train.PolicyValueNetwork(net_key, grid_size=4)
    grid = jnp.zeros((4, 4), dtype=jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    states = jax.vmap(game.create_initial_state)(jnp.broadcast_to(grid, (args.num_envs, 4, 4)))
    # Mature stacks and a near-timeout state ensure both moves and auto-resets are exercised.
    states = states._replace(time=jnp.arange(args.num_envs, dtype=jnp.int32) % 2 * 498, armies=states.armies * 10)
    jax.block_until_ready((states, network))

    def python_rollout(states, network, key):
        data = []
        for _ in range(args.steps):
            states, row, key = train.rollout_step(states, network, key)
            data.append(row)
        return states, jax.tree.map(lambda *xs: jnp.stack(xs), *data), key

    @eqx.filter_jit
    def scan_rollout(states, network, key):
        def body(carry, _):
            states, row, key = train.rollout_step(carry[0], network, carry[1])
            return (states, key), row

        (states, key), data = jax.lax.scan(body, (states, key), None, length=args.steps)
        return states, data, key

    @eqx.filter_jit
    def inference(states, network, key):
        observations = jax.vmap(lambda state: game.get_observation(state, 0))(states)
        arrays = jax.vmap(train.obs_to_array)(observations)
        masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(observations)
        return jax.vmap(network, in_axes=(0, 0, 0, None))(arrays, masks, jax.random.split(key, args.num_envs), None)

    @eqx.filter_jit
    def prepare_batch(final_states, network, data):
        obs, masks, actions, logprobs, values, rewards, dones, _ = data
        observations = jax.vmap(lambda state: game.get_observation(state, 0))(final_states)
        arrays = jax.vmap(train.obs_to_array)(observations)
        final_masks = jax.vmap(lambda o: compute_valid_move_mask(o.armies, o.owned_cells, o.mountains))(observations)
        action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        bootstrap = jax.vmap(lambda o, m: network(o, m, None, action)[1])(arrays, final_masks)
        advantages = train.compute_gae(rewards, values, dones, bootstrap)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return obs, masks, actions, logprobs, advantages, returns

    measure("rollout_step", train.rollout_step, states, network, key, units=args.num_envs)
    original = measure(
        "python_rollout_and_stack", python_rollout, states, network, key, units=args.num_envs * args.steps
    )
    scanned = measure("scan_rollout", scan_rollout, states, network, key, units=args.num_envs * args.steps)
    integer_exact = True
    float_close = True
    max_absolute_difference = 0.0
    for left, right in zip(jax.tree.leaves(original), jax.tree.leaves(scanned), strict=True):
        left, right = np.asarray(left), np.asarray(right)
        if np.issubdtype(left.dtype, np.inexact):
            float_close &= bool(np.allclose(left, right, atol=1e-5, rtol=1e-5))
            max_absolute_difference = max(max_absolute_difference, float(np.max(np.abs(left - right), initial=0)))
        else:
            integer_exact &= bool(np.array_equal(left, right))
    report["correctness"] = {
        "integer_and_boolean_leaves_exact": integer_exact,
        "floating_leaves_close_atol_rtol_1e_5": float_close,
        "max_absolute_float_difference": max_absolute_difference,
        "covers": "All final state, PRNG key, observation, mask, action, logprob, value, reward, done and info leaves",
        "episode_end_count": int(jnp.sum(original[1][6])),
    }
    print(f"correctness: {json.dumps(report['correctness'])}", flush=True)
    measure("observation_mask_and_inference", inference, states, network, key, units=args.num_envs)
    batch = measure("bootstrap_and_gae", prepare_batch, original[0], network, original[1])
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    measure("optimization", train.train_step, network, opt_state, batch, optimizer, units=args.num_envs * args.steps)

    def iteration(rollout):
        final, data, _ = rollout(states, network, key)
        batch = prepare_batch(final, network, data)
        return train.train_step(network, opt_state, batch, optimizer)

    measure("python_iteration", lambda: iteration(python_rollout), units=args.num_envs * args.steps)
    measure("scan_iteration", lambda: iteration(scan_rollout), units=args.num_envs * args.steps)
    report["scan_vs_python_rollout_speedup"] = (
        report["timings"]["python_rollout_and_stack"]["median_seconds"]
        / report["timings"]["scan_rollout"]["median_seconds"]
    )
    report["scan_vs_python_iteration_speedup"] = (
        report["timings"]["python_iteration"]["median_seconds"] / report["timings"]["scan_iteration"]["median_seconds"]
    )
    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.trace)):
            for name, rollout in (("python", python_rollout), ("scan", scan_rollout)):
                with jax.profiler.TraceAnnotation(name):
                    jax.block_until_ready(iteration(rollout))
    with contextlib.suppress(RuntimeError):
        report["device_memory_stats"] = jax.devices()[0].memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not integer_exact or not float_close:
        raise SystemExit("Rollout equivalence failed; report saved, do not accept performance result")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

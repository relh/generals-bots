"""Compare reference and compiled PPO updates on one frozen training fixture."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Frozen completed-run checkpoint; generates one real rollout")
    source.add_argument("--fixture", type=Path, help="Reuse the fixture saved by an earlier invocation")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trace", type=Path)
    parser.add_argument(
        "--with-rollout", action="store_true", help="Also time rollout+GAE+optimization (requires checkpoint)"
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive; warmup must be nonnegative")
    if args.with_rollout and not args.checkpoint:
        parser.error("--with-rollout requires --checkpoint")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.device == "cpu" else "cuda"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    sys.path.insert(0, str(ROOT))
    import equinox as eqx
    import jax
    import numpy as np
    import optax

    from generals.training.checkpoint import load_checkpoint, save_checkpoint
    from generals.training.optimization import optimize, summarize
    from generals.training.ppo import Batch, compute_gae, update
    from generals.training.train import Config, make_env, make_rollout, opponents_for

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        fixture = load_checkpoint(args.fixture)
    else:
        snapshot = load_checkpoint(args.checkpoint)
        config = Config(**snapshot["config"])
        env = make_env(config, snapshot["stage"])
        env.pool_size = snapshot["pool"].armies.shape[0]
        rollout = make_rollout(env, opponents_for(config, env), config)

        def collect_batch():
            carry, data = rollout(
                snapshot["network"],
                snapshot["pool"],
                snapshot["states"],
                snapshot["sides"],
                snapshot["opponent_ids"],
                snapshot["episode_returns"],
                snapshot["rng"],
            )
            jax.block_until_ready(data)
            advantages = compute_gae(
                data["rewards"],
                data["values"],
                data["next_values"],
                data["bootstrap_terminal"],
                data["truncated"],
                config.gamma,
                config.gae_lambda,
            )
            returns = advantages + data["values"]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            n = config.num_envs * config.steps
            batch = Batch(
                *(
                    x.reshape(n, *x.shape[2:])
                    for x in (
                        data["observations"],
                        data["masks"],
                        data["actions"],
                        data["logprobs"],
                        advantages,
                        returns,
                    )
                )
            )
            return batch, carry[-1]

        batch, rollout_key = collect_batch()
        fixture = dict(
            network=snapshot["network"],
            optimizer=snapshot["optimizer"],
            batch=batch,
            rng=rollout_key,
            config=snapshot["config"],
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        )
    config = Config(**fixture["config"])
    fixture_path = args.output.with_suffix(".fixture.pkl")
    save_checkpoint(fixture_path, fixture)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate))
    n = fixture["batch"].observations.shape[0]
    arguments = (fixture["network"], fixture["optimizer"], fixture["batch"], optimizer, fixture["rng"])
    jax.block_until_ready((fixture["network"], fixture["optimizer"], fixture["batch"], fixture["rng"]))
    options = dict(
        epochs=config.epochs,
        minibatch_size=config.minibatch_size,
        clip=config.clip,
        entropy_weight=config.entropy_weight,
        target_kl=config.target_kl,
    )

    def reference(network, state, batch, optimizer, key):
        diagnostics = []
        early = False
        for _ in range(config.epochs):
            key, shuffle = jax.random.split(key)
            indices = jax.random.permutation(shuffle, n)
            for begin in range(0, n, config.minibatch_size):
                minibatch = jax.tree.map(lambda x: x[indices[begin : begin + config.minibatch_size]], batch)
                network, state, row = update(network, state, minibatch, optimizer, config.clip, config.entropy_weight)
                row = {name: float(value) for name, value in row.items()}
                if not all(np.isfinite(value) for value in row.values()):
                    raise FloatingPointError(f"nonfinite optimizer diagnostic: {row}")
                diagnostics.append(row)
                if row["approx_kl"] > config.target_kl:
                    early = True
                    break
            if early:
                break
        means = {name: float(np.mean([row[name] for row in diagnostics])) for name in diagnostics[0]}
        return network, state, key, means, len(diagnostics), early

    def compiled(*arguments):
        result = optimize(*arguments, **options)
        means, count, early = summarize(result)
        return result.network, result.optimizer_state, result.key, means, count, early

    report = dict(
        command=sys.argv,
        config=fixture["config"],
        fixture=str(fixture_path),
        fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        checkpoint_sha256=fixture.get("checkpoint_sha256"),
        runtime=dict(
            platform=platform.platform(),
            python=sys.version,
            jax=jax.__version__,
            equinox=eqx.__version__,
            optax=optax.__version__,
            devices=[str(d) for d in jax.devices()],
            device_kinds=[d.device_kind for d in jax.devices()],
            cpu_affinity=sorted(os.sched_getaffinity(0)),
            environment={
                name: os.environ.get(name)
                for name in ("JAX_PLATFORMS", "LD_LIBRARY_PATH", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE")
            },
        ),
        sources={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(),
                ROOT / "generals/training/optimization.py",
                ROOT / "generals/training/ppo.py",
                ROOT / "generals/training/train.py",
            )
        },
        semantics=(
            "Same immutable network/optimizer/RNG/batch for every call. Timings include host diagnostic means "
            "and synchronize all outputs; exclude rollout, GAE, checkpoints and episode logging. "
            "First call includes compilation; warmed paired order alternates."
        ),
        timings={},
    )
    results = {}
    # Warm both implementations before alternating timed order to limit drift.
    for name, function in (("reference", reference), ("compiled", compiled)):
        start = time.perf_counter()
        results[name] = function(*arguments)
        jax.block_until_ready(results[name])
        first = time.perf_counter() - start
        for _ in range(args.warmup):
            jax.block_until_ready(function(*arguments))
        report["timings"][name] = dict(first_call_seconds=first, seconds=[])
        print(f"{name} first call: {first:.3f}s", flush=True)
    for repeat in range(args.repeats):
        order = [("reference", reference), ("compiled", compiled)]
        if repeat % 2:
            order.reverse()
        for name, function in order:
            start = time.perf_counter()
            results[name] = function(*arguments)
            jax.block_until_ready(results[name])
            report["timings"][name]["seconds"].append(time.perf_counter() - start)
    for name, values in report["timings"].items():
        values["median_seconds"] = statistics.median(values["seconds"])
        values["min_seconds"], values["max_seconds"] = min(values["seconds"]), max(values["seconds"])
        values["environment_steps_per_optimization_second"] = n / values["median_seconds"]
        print(f"{name}: {json.dumps(values)}", flush=True)
    left, right = results["reference"], results["compiled"]
    exact_integers = True
    close_floats = True
    maximum_difference = 0.0
    for a, b in zip(jax.tree.leaves(left[:3]), jax.tree.leaves(right[:3]), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if np.issubdtype(a.dtype, np.inexact):
            close_floats &= bool(np.allclose(a, b, atol=2e-6, rtol=2e-5))
            maximum_difference = max(maximum_difference, float(np.max(np.abs(a - b), initial=0)))
        else:
            exact_integers &= bool(np.array_equal(a, b))
    stats_close = all(np.isclose(left[3][name], right[3][name], atol=2e-6, rtol=2e-5) for name in left[3])
    report["parity"] = dict(
        integer_optimizer_state_and_rng_exact=exact_integers,
        floating_parameters_and_optimizer_close=close_floats,
        max_absolute_parameter_or_optimizer_difference=maximum_difference,
        diagnostics_close=bool(stats_close),
        reference_diagnostics=left[3],
        compiled_diagnostics=right[3],
        reference_update_count=left[4],
        compiled_update_count=right[4],
        reference_early_stop=left[5],
        compiled_early_stop=right[5],
    )
    report["speedup"] = (
        report["timings"]["reference"]["median_seconds"] / report["timings"]["compiled"]["median_seconds"]
    )
    if args.with_rollout:
        iteration_times = {name: [] for name in ("reference", "compiled")}
        for repeat in range(args.warmup + args.repeats):
            order = [("reference", reference), ("compiled", compiled)]
            if repeat % 2:
                order.reverse()
            for name, function in order:
                start = time.perf_counter()
                batch, key = collect_batch()
                result = function(fixture["network"], fixture["optimizer"], batch, optimizer, key)
                jax.block_until_ready(result)
                if repeat >= args.warmup:
                    iteration_times[name].append(time.perf_counter() - start)
        report["iteration_timings"] = {
            name: dict(
                seconds=seconds,
                median_seconds=statistics.median(seconds),
                environment_steps_per_second=n / statistics.median(seconds),
            )
            for name, seconds in iteration_times.items()
        }
        report["iteration_speedup"] = statistics.median(iteration_times["reference"]) / statistics.median(
            iteration_times["compiled"]
        )
        report["iteration_semantics"] = (
            "Same immutable checkpoint states, pool, policy and RNG. Includes rollout, GAE and optimizer "
            "with diagnostic means; excludes map refresh, episode JSON logging and checkpoint writes."
        )
        print(f"iteration: {json.dumps(report['iteration_timings'])}", flush=True)
    if args.trace:
        args.trace.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(str(args.trace)):
            for name, function in (("reference", reference), ("compiled", compiled)):
                with jax.profiler.TraceAnnotation(name):
                    jax.block_until_ready(function(*arguments))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not (exact_integers and close_floats and stats_close and left[4:] == right[4:]):
        raise SystemExit(f"Parity failed; report saved to {args.output}")
    print(f"Parity passed; speedup={report['speedup']:.3f}x; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

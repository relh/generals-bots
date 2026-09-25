"""Time the GPU and Python parts of one batched Classic environment hot loop."""

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import jax
import numpy as np

from metta_training.environment import EnvironmentContext, NativeEnvironment
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--transport", action="store_true")
    parser.add_argument("--lean", action="store_true")
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("B300 CUDA device required")
    options = dict(json.loads(args.build.read_text())["config"]["python_environment"]["options"])
    options["parallel_games"] = args.games
    if args.lean:
        options["compact_features"] = True
        options["lean_features"] = True
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1104, index=0, mode="train", output=Path("/tmp")),
        **options,
    )
    obs = env.reset("profile-classic-1104")
    original_advance = env._advance_states
    original_observation = env._observation
    kernel_seconds = []
    observation_seconds = []

    def timed_advance(*inputs):
        start = time.perf_counter()
        result = original_advance(*inputs)
        jax.tree.map(lambda value: value.block_until_ready() if hasattr(value, "block_until_ready") else value, result)
        kernel_seconds.append(time.perf_counter() - start)
        return result

    def timed_observation(*inputs):
        start = time.perf_counter()
        result = original_observation(*inputs)
        observation_seconds.append(time.perf_counter() - start)
        return result

    env._advance_states = timed_advance
    env._observation = timed_observation
    serializer = NativeEnvironment.__new__(NativeEnvironment)
    serializer.spec = env.spec
    profiler = cProfile.Profile()
    totals = []
    transport_seconds = []
    for turn in range(args.warmup + args.steps):
        # Choose the first legal move, once movement is possible; this choice
        # happens outside the measured environment step.
        legal = np.asarray(obs.action_masks, dtype=bool)[:, :env.spec.action_sizes[0]]
        index = np.argmax(legal, axis=1)
        actions = np.stack((index, np.zeros(args.games, dtype=np.int32)), axis=1).tolist()
        start = time.perf_counter()
        if turn == args.warmup:
            profiler.enable()
        obs = env.step(actions).observation
        if args.transport:
            before_transport = time.perf_counter()
            serializer.encode(obs)
            if turn >= args.warmup:
                transport_seconds.append(time.perf_counter() - before_transport)
        if turn >= args.warmup:
            totals.append(time.perf_counter() - start)
    profiler.disable()
    env.close()
    kernel = np.asarray(kernel_seconds[args.warmup:])
    observation = np.asarray(observation_seconds[args.warmup:])
    total = np.asarray(totals)
    print(json.dumps({
        "games": args.games, "sampled_steps": args.steps,
        "observation_size": env.spec.observation_size,
        "total_ms_median": float(np.median(total) * 1000),
        "kernel_ms_median": float(np.median(kernel) * 1000),
        "numeric_observation_ms_median": float(np.median(observation) * 1000),
        "other_ms_median": float(np.median((total - kernel - observation) * 1000)),
        "total_ms_p90": float(np.percentile(total, 90) * 1000),
        "transport_ms_median": float(np.median(transport_seconds) * 1000) if transport_seconds else None,
    }), flush=True)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(22)
    print(stream.getvalue(), flush=True)


if __name__ == "__main__":
    main()

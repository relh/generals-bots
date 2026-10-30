"""Time the current Classic GPU environment and its teacher/transport stages."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def measure(name, call, warmup, steps, games):
    times = []
    for index in range(warmup + steps):
        start = time.perf_counter()
        jax.block_until_ready(call())
        if index >= warmup:
            times.append(time.perf_counter() - start)
    median = float(np.median(times))
    print(json.dumps({
        "stage": name,
        "games": games,
        "warmup": warmup,
        "steps": steps,
        "median_ms": median * 1000,
        "p90_ms": float(np.percentile(times, 90) * 1000),
        "stage_sps": games / median,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("CUDA GPU required")
    options = dict(json.loads(args.build.read_text())["config"]["python_environment"]["options"])
    options["parallel_games"] = args.games
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1205, index=0, mode="train", output=Path("/tmp")),
        **options,
    )
    try:
        _, masks = env.reset_device("pipeline-profile-1205")
        keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(env.keys)
        teacher_actions = env._teacher_actions(env.states, env.sides, keys)
        values, raw_masks = env._observe_states(env.states, env.sides)
        measure("teacher", lambda: env._teacher_actions(env.states, env.sides, keys),
                args.warmup, args.steps, args.games)
        measure("observe", lambda: env._observe_states(env.states, env.sides),
                args.warmup, args.steps, args.games)
        measure("transport", lambda: env._device_transport(values, raw_masks, teacher_actions),
                args.warmup, args.steps, args.games)

        def full_step():
            nonlocal masks
            indices = jnp.argmax(masks[:, : env.spec.action_sizes[0]], axis=1)
            actions = jnp.stack((indices, jnp.zeros_like(indices)), axis=1)
            result = env.step_device(actions)
            masks = result[1]
            return result[:4]

        measure("full_step", full_step, args.warmup, args.steps, args.games)
    finally:
        env.close()


if __name__ == "__main__":
    main()

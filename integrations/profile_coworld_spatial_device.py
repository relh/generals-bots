"""Measure full Classic device steps with the spatial pilot's exact environment."""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    options = dict(json.loads(args.build.read_text())["config"]["python_environment"]["options"])
    assert options["coworld_classic"] and options["land_gain_reward_weight"] == 0.2
    assert options["teacher"] is None and options["shaping_gamma"] == 0.999
    options["parallel_games"] = args.games
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=691, index=0, mode="train", output=Path("/recovery")),
        **options,
    )
    try:
        _, masks = env.reset_device("spatial-device-profile-691")
        durations = []
        for step in range(args.warmup + args.steps):
            start = time.perf_counter()
            indices = jnp.argmax(masks[:, : env.spec.action_sizes[0]], axis=1)
            actions = jnp.stack((indices, jnp.zeros_like(indices)), axis=1)
            values, masks, rewards, done, episode_done = env.step_device(actions)
            jax.block_until_ready((values, masks, rewards, done))
            duration = time.perf_counter() - start
            if step >= args.warmup:
                durations.append(duration)
                print(json.dumps({"step": step + 1, "ms": round(duration * 1000, 3),
                                  "pool_generation": env._pool_generation}), flush=True)
        total = sum(durations)
        print(json.dumps({"games": args.games, "warmup": args.warmup, "steps": args.steps,
                          "total_seconds": total, "end_to_end_env_sps": args.games * args.steps / total,
                          "median_ms": float(np.median(durations)) * 1000,
                          "p90_ms": float(np.percentile(durations, 90)) * 1000,
                          "max_ms": max(durations) * 1000}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

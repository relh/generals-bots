"""Measure the full Classic JAX step with actions and outputs kept on device."""

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
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("B300 CUDA device required")
    options = dict(json.loads(args.build.read_text())["config"]["python_environment"]["options"])
    options["parallel_games"] = args.games
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=1104, index=0, mode="train", output=Path("/tmp")),
        **options,
    )
    try:
        env.reset("device-profile-classic-1104")
        head_size = env.spec.action_sizes[0]
        action_mask = env._observe_states(env.states, env.sides)[1]
        move_boundary = (4 if env.base.factorized_actions else 8) * env.base.size**2
        cached_teacher_actions = jnp.zeros((args.games, 5), dtype=jnp.int32)
        alive = jnp.ones(args.games, dtype=bool)
        samples = []
        measured_indices = []
        measured_done = []
        for turn in range(args.warmup + args.steps):
            start = time.perf_counter()
            indices = jnp.argmax(action_mask[:, :head_size], axis=1)
            splits = jnp.zeros_like(indices)
            result = env._advance_states(
                env.states, env.base.pool, env.sides, env.opponent_ids,
                indices, splits, env.keys, cached_teacher_actions, alive,
            )
            jax.block_until_ready(result)
            env.states, env.keys, _, action_mask, _, _, _, _ = result
            if turn >= args.warmup:
                samples.append(time.perf_counter() - start)
                measured_indices.append(indices)
                measured_done.append(result[5])
        times = np.asarray(samples)
        median = float(np.median(times))
        move_fraction = float(jnp.mean(jnp.stack(measured_indices) < move_boundary))
        terminal_fraction = float(jnp.mean(jnp.stack(measured_done)))
        print(json.dumps({
            "games": args.games, "warmup_steps": args.warmup,
            "measured_steps": args.steps, "device_step_ms_median": median * 1000,
            "device_step_ms_p90": float(np.percentile(times, 90) * 1000),
            "early_200_step_ms_median": float(np.median(times[:200]) * 1000),
            "late_200_step_ms_median": float(np.median(times[-200:]) * 1000),
            "device_only_rollout_sps": args.games / median,
            "move_fraction": move_fraction,
            "terminal_fraction": terminal_fraction,
        }), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

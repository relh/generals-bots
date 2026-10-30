"""Bounded CUDA rollout benchmark for the Generals Puffer environment.

This measures environment.step with the actual training observation and teacher
path. It excludes policy inference and optimization, so its SPS is an upper
bound on end-to-end Puffer training throughput.
"""

import argparse
import json
import time
from pathlib import Path

import jax
from metta_training.environment import EnvironmentContext, NativeEnvironment

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-games", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=32)
    parser.add_argument("--native-encode", action="store_true")
    args = parser.parse_args()
    if jax.devices()[0] not in jax.devices("cuda"):
        raise RuntimeError("This benchmark requires CUDA")

    for count in args.parallel_games:
        env = BatchedGeneralsPufferEnvironment(
            context=EnvironmentContext(seed=241, index=0, mode="train", output=Path.cwd()),
            board_size=10,
            horizon=300,
            opponent="mixed",
            shaping_weight=1.0,
            teacher="harvester",
            supervise_teacher=True,
            factorized_actions=True,
            parallel_games=count,
        )
        actions = [[400, 0] for _ in range(count)]
        started = time.perf_counter()
        env.reset(f"bench:{count}")
        for _ in range(args.warmup_steps):
            env.step(actions)
        compiled_seconds = time.perf_counter() - started
        components = {"advance": 0.0, "teacher": 0.0, "observation": 0.0}
        advance = env._advance_states
        teacher = env._teacher_actions
        observation = env._observation

        def timed_advance(*values):
            start = time.perf_counter()
            result = advance(*values)
            result[5].block_until_ready()
            components["advance"] += time.perf_counter() - start
            return result

        def timed_teacher(*values):
            start = time.perf_counter()
            result = teacher(*values)
            result.block_until_ready()
            components["teacher"] += time.perf_counter() - start
            return result

        def timed_observation(*values):
            start = time.perf_counter()
            result = observation(*values)
            components["observation"] += time.perf_counter() - start
            return result

        env._advance_states = timed_advance
        env._teacher_actions = timed_teacher
        env._observation = timed_observation
        native = NativeEnvironment.__new__(NativeEnvironment)
        native.spec = env.spec
        encoded_seconds = 0.0
        started = time.perf_counter()
        for _ in range(args.measure_steps):
            transition = env.step(actions)
            if args.native_encode:
                encode_started = time.perf_counter()
                native.encode(transition.observation)
                encoded_seconds += time.perf_counter() - encode_started
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "device": jax.devices("cuda")[0].device_kind,
                    "parallel_games": count,
                    "native_encode": args.native_encode,
                    "encoded_seconds": encoded_seconds,
                    "warmup_steps": args.warmup_steps,
                    "measure_steps": args.measure_steps,
                    "warmup_seconds": compiled_seconds,
                    "measure_seconds": elapsed,
                    "rollout_sps_upper_bound": count * args.measure_steps / elapsed,
                    "component_seconds": components,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        env.close()


if __name__ == "__main__":
    main()

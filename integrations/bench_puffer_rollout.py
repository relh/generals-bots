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
import numpy as np
from metta_training.environment import EnvironmentContext, NativeEnvironment

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel-games", type=int, nargs="+")
    parser.add_argument("--build-config", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=32)
    parser.add_argument("--native-encode", action="store_true")
    parser.add_argument("--teacher-actions", action="store_true")
    args = parser.parse_args()
    if jax.devices()[0] not in jax.devices("cuda"):
        raise RuntimeError("This benchmark requires CUDA")

    configured = json.loads(args.build_config.read_text())["python_environment"]["options"] if args.build_config else None
    counts = args.parallel_games or ([configured["parallel_games"]] if configured else [16, 64, 256, 1024])
    for count in counts:
        options = dict(configured) if configured else dict(
            board_size=10, horizon=300, opponent="mixed", shaping_weight=1.0,
            teacher="harvester", supervise_teacher=True, factorized_actions=True,
        )
        options["parallel_games"] = count
        env = BatchedGeneralsPufferEnvironment(
            context=EnvironmentContext(seed=241, index=0, mode="train", output=Path.cwd()),
            **options,
        )
        actions = [[4 * env.base.size**2, 0] for _ in range(count)]
        teacher = getattr(env, "_teacher_actions", None)
        if args.teacher_actions and teacher is None:
            raise ValueError("Teacher actions require a supervised teacher build")

        def next_actions():
            if not args.teacher_actions:
                return actions
            keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(env.keys)
            chosen = np.asarray(teacher(env.states, env.sides, keys))
            cells = env.base.size**2
            moves = chosen[:, 3] * cells + chosen[:, 1] * env.base.size + chosen[:, 2]
            indices = np.where(chosen[:, 0] == 1, 4 * cells, moves)
            return np.stack((indices, chosen[:, 4]), axis=1).tolist()

        started = time.perf_counter()
        env.reset(f"bench:{count}")
        for _ in range(args.warmup_steps):
            env.step(next_actions())
        compiled_seconds = time.perf_counter() - started
        components = {"advance": 0.0, "teacher": 0.0, "observation": 0.0}
        advance = env._advance_states
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
        if teacher is not None:
            env._teacher_actions = timed_teacher
        env._observation = timed_observation
        native = NativeEnvironment.__new__(NativeEnvironment)
        native.spec = env.spec
        encoded_seconds = 0.0
        elapsed = 0.0
        for _ in range(args.measure_steps):
            active_actions = next_actions()
            started = time.perf_counter()
            transition = env.step(active_actions)
            if args.native_encode:
                encode_started = time.perf_counter()
                native.encode(transition.observation)
                encoded_seconds += time.perf_counter() - encode_started
            elapsed += time.perf_counter() - started
        print(
            json.dumps(
                {
                    "device": jax.devices("cuda")[0].device_kind,
                    "parallel_games": count,
                    "native_encode": args.native_encode,
                    "teacher_actions": args.teacher_actions,
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

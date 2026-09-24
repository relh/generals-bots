"""Time Coworld Classic JAX reset and one batched step on CUDA."""

import time
import sys
import cProfile
import io
import pstats
from pathlib import Path

import jax

from metta_training.environment import EnvironmentContext, NativeEnvironment
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    if not jax.devices("cuda"):
        raise RuntimeError("Environment probe requires CUDA")
    context = EnvironmentContext(seed=603, index=0, mode="train", output=Path("/tmp"))
    games = int(sys.argv[1])
    features = sys.argv[2] == "features"
    compact = sys.argv[2] == "compact"
    teacher = sys.argv[3] == "teacher"
    with_transport = len(sys.argv) > 4 and sys.argv[4] == "transport"
    started = time.monotonic()
    env = BatchedGeneralsPufferEnvironment(
        context=context, parallel_games=games, coworld_classic=True, coworld_pool_size=64,
        opponent="mixed", teacher="harvester" if teacher else None, supervise_teacher=teacher,
        factorized_actions=True, goal_features=features, compact_features=compact, shaping_weight=1.0,
    )
    print(f"construct_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    env.reset("603:0:0")
    print(f"reset_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    action = [env.spec.action_sizes[0] - 1, 0]
    result = env.step([action] * env.parallel_games)
    print(f"first_step_seconds={time.monotonic() - started:.3f} rewards={len(result.rewards)}", flush=True)
    timings = []
    observation_times = []
    transport_times = []
    serializer = NativeEnvironment.__new__(NativeEnvironment)
    serializer.spec = env.spec
    original_observation = env._observation

    def timed_observation(*args):
        started = time.monotonic()
        value = original_observation(*args)
        observation_times.append(time.monotonic() - started)
        return value

    env._observation = timed_observation
    for _ in range(8):
        started = time.monotonic()
        transition = env.step([action] * env.parallel_games)
        timings.append(time.monotonic() - started)
        if with_transport:
            started = time.monotonic()
            serializer.encode(transition.observation)
            transport_times.append(time.monotonic() - started)
    steady = sum(timings[2:]) / len(timings[2:])
    observed = sum(observation_times[2:]) / len(observation_times[2:])
    print(f"warm_step_seconds={steady:.4f} observation_seconds={observed:.4f} "
          f"rollout_sps={games / steady:.0f}", flush=True)
    if transport_times:
        encoded = sum(transport_times[2:]) / len(transport_times[2:])
        print(f"transport_seconds={encoded:.4f} combined_sps={games / (steady + encoded):.0f}", flush=True)
        profile = cProfile.Profile()
        profile.runcall(serializer.encode, transition.observation)
        report = io.StringIO()
        pstats.Stats(profile, stream=report).sort_stats("cumtime").print_stats(20)
        print(report.getvalue(), flush=True)


if __name__ == "__main__":
    main()

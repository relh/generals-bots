"""Measure teacher-label coverage in one seeded CUDA Generals rollout.

Run with PYTHONPATH pointing at the chosen adapter source and CUDA JAX runtime.
The same script measures frozen-seat and recycling variants without changing
their code. It uses Harvester actions so both variants see the same curriculum.
"""

import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    device = jax.devices()[0]
    if device not in jax.devices("cuda"):
        raise RuntimeError(f"Expected a CUDA JAX device, got {device}")

    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=241, index=0, mode="train", output=Path.cwd()),
        board_size=10,
        horizon=300,
        opponent="mixed",
        shaping_weight=1.0,
        teacher="harvester",
        supervise_teacher=True,
        factorized_actions=True,
        parallel_games=16,
    )
    env.reset("241:0:0")
    labels = 0
    for _ in range(300):
        keys = jax.vmap(lambda key: jax.random.fold_in(jax.random.split(key)[0], 37))(env.keys)
        actions = np.asarray(env._teacher_actions(env.states, env.sides, keys))
        index = np.where(actions[:, 0] == 1, 400, actions[:, 3] * 100 + actions[:, 1] * 10 + actions[:, 2])
        transition = env.step(np.stack((index, actions[:, 4]), axis=1).tolist())
        labels += sum(int(target.weights[0]) for target in transition.observation.teachers)

    lives = getattr(env, "completed", env.finished)
    print(
        json.dumps(
            {
                "device": str(device),
                "seed": 241,
                "turns": 300,
                "parallel_games": 16,
                "labeled_rows": labels,
                "total_rows": 4800,
                "label_fraction": labels / 4800,
                "completed_lives": int(lives.sum()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()

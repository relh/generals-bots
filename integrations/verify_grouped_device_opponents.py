"""Check GPU transition parity before timing grouped-opponent Puffer training."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    options = json.loads(args.build.read_text())["config"]["python_environment"]["options"]
    assert options.pop("group_device_opponents")
    options.update(parallel_games=32, coworld_pool_size=16)
    context = EnvironmentContext(seed=1326, index=0, mode="train", output=args.output.parent)
    baseline = BatchedGeneralsPufferEnvironment(context=context, **options)
    grouped = BatchedGeneralsPufferEnvironment(context=context, group_device_opponents=True, **options)
    try:
        first, second = baseline.reset_device("1326"), grouped.reset_device("1326")
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        moved = False
        recycled = 0
        for step in range(12):
            if step in (4, 9):
                resetting = jnp.arange(32) % 2 == step // 5
                for env in (baseline, grouped):
                    env.states = env.states._replace(time=jnp.where(resetting, env.horizon - 1, env.states.time))
            moves = np.argmax(np.asarray(first[1])[:, :1765], axis=1)
            moved |= bool(np.any(moves != 1764))
            actions = jnp.asarray(np.stack((moves, np.full(32, step % 2)), axis=1), dtype=jnp.float32)
            first, second = baseline.step_device(actions), grouped.step_device(actions)
            recycled += int(np.asarray(first[3]).sum())
            for a, b in zip(first, second, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree.leaves(baseline.states), jax.tree.leaves(grouped.states), strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            np.testing.assert_array_equal(np.asarray(baseline.keys), np.asarray(grouped.keys))
        assert moved and recycled >= 32
    finally:
        baseline.close()
        grouped.close()
    result = dict(transitions=384, recycled_lanes=recycled, device=str(jax.devices()[0]), exact_parity=True)
    args.output.write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

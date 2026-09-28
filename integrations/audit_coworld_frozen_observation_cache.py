"""Compare cached frozen-opponent observations with fresh GPU encoding."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsFrozenOpponentPufferEnvironment


def same(left, right):
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def verify_cache(environment):
    encoded = environment._observe_both(environment.states)
    assert same(environment._cached_values, encoded[0])
    assert same(environment._cached_masks, encoded[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    build = json.loads(args.build.read_text())
    assert build["config"]["python_environment"]["factory"].endswith(":BatchedGeneralsFrozenOpponentPufferEnvironment")
    options = dict(build["config"]["python_environment"]["options"])
    options.update(parallel_games=16, coworld_pool_size=16, horizon=8)
    args.output.mkdir(parents=True, exist_ok=False)
    cached = BatchedGeneralsFrozenOpponentPufferEnvironment(
        context=EnvironmentContext(seed=1729, index=0, mode="train", output=args.output / "cached"),
        **options,
    )
    reference = BatchedGeneralsFrozenOpponentPufferEnvironment(
        context=EnvironmentContext(seed=1729, index=0, mode="train", output=args.output / "reference"),
        **options,
    )
    try:
        observed = cached.reset_device("1729:0:0")
        expected = reference.reset_device("1729:0:0")
        assert same(observed, expected)
        assert same(cached.states, reference.states)
        verify_cache(cached)
        for turn in range(12):
            masks = np.asarray(observed[1], dtype=bool)
            rng = np.random.default_rng(1729 + turn)
            moves = cached.spec.action_sizes[0]
            actions = np.stack(
                [
                    [
                        int(rng.choice(np.flatnonzero(masks[row, :moves]))),
                        int(rng.choice(np.flatnonzero(masks[row, moves:]))),
                    ]
                    for row in range(16)
                ]
            ).astype(np.int32)
            reference._cached_values, reference._cached_masks = reference._observe_both(reference.states)
            cached_step = cached.step_device(jnp.asarray(actions))
            reference_step = reference.step_device(jnp.asarray(actions))
            assert same(cached_step, reference_step), f"transition mismatch on turn {turn}"
            assert same(cached.states, reference.states), f"state mismatch on turn {turn}"
            verify_cache(cached)
            observed = cached_step[:2]
        result = {"games": 16, "turns": 12, "pool_refreshes": 1, "observation_and_transition_parity": True}
        (args.output / "cache-audit.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        cached.close()
        reference.close()


if __name__ == "__main__":
    main()

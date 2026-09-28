"""Compare native frozen-opponent dispatch with real two-seat game transitions."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.native_selfplay import NativeFrozenOpponentPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.require_gpu and jax.devices()[0].platform != "gpu":
        raise RuntimeError("GPU transition audit requested")
    options = json.loads(args.build.read_text())["config"]["python_environment"]["options"]
    options.update(parallel_games=8, coworld_pool_size=64, require_gpu=args.require_gpu)
    context = EnvironmentContext(seed=4081, index=0, mode="train", output=args.output.parent)
    frozen = NativeFrozenOpponentPufferEnvironment(
        frozen_build=args.build, frozen_training=args.training, frozen_checkpoint=args.checkpoint,
        frozen_sha256=args.sha256, context=context, **options,
    )
    reference = BatchedGeneralsSelfPlayPufferEnvironment(context=context, **options)
    rows = jnp.arange(8)
    carry = frozen._frozen.initial_state(8)
    recycled = 0
    try:
        frozen.reset_device("4081")
        reference.reset_device("4081")
        assert np.array_equal(frozen.sides, reference.sides)
        for tick in range(32):
            if tick == 3:
                # Force two real selective truncations, keeping all other games live.
                for env in (frozen, reference):
                    env.states = env.states._replace(
                        time=env.states.time.at[jnp.array([2, 5])].set(env.horizon - 1)
                    )
                frozen._cached_values, frozen._cached_masks = frozen._observe_both(frozen.states)
            values, masks = reference._observe_both(reference.states)
            side = reference.sides
            learner = jnp.argmax(masks[rows, side], axis=-1)[:, None]
            if tick % 2:
                learner = jnp.full((8, 1), 3528, jnp.int32)
            opposing, next_carry = frozen._frozen.actions(
                values[rows, 1 - side], masks[rows, 1 - side], carry
            )
            predictions, _ = frozen._frozen.forward(values[rows, 1 - side], carry)
            assert np.isfinite(np.asarray(predictions)).all()
            assert np.all(np.asarray(masks[rows, 1 - side, opposing[:, 0]]))
            paired = jnp.zeros((8, 2, 1), jnp.int32)
            paired = paired.at[rows, side].set(learner)
            paired = paired.at[rows, 1 - side].set(opposing)
            actual = frozen.step_device(learner)
            expected = reference.step_device(paired.reshape(16, 1))
            ev = expected[0].reshape(8, 2, -1)[rows, side]
            em = expected[1].reshape(8, 2, -1)[rows, side]
            er = expected[2].reshape(8, 2)[rows, side]
            ed = expected[3].reshape(8, 2)[:, 0]
            for lhs, rhs in zip(actual[:4], (ev, em, er, ed), strict=True):
                np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))
            for lhs, rhs in zip(frozen.states, reference.states, strict=True):
                np.testing.assert_array_equal(np.asarray(lhs), np.asarray(rhs))
            carry = jnp.where(ed[None, :, None].astype(bool), 0, next_carry)
            np.testing.assert_array_equal(np.asarray(frozen._frozen_state), np.asarray(carry))
            assert np.isfinite(np.asarray(carry)).all()
            recycled += int(np.asarray(ed).sum())
            if tick == 3:
                assert np.asarray(ed)[[2, 5]].all()
                assert not np.asarray(carry)[:, [2, 5]].any()
        report = dict(
            platform=jax.devices()[0].platform, games=8, transitions=256,
            learner_player0=int(np.asarray(frozen.sides == 0).sum()),
            learner_player1=int(np.asarray(frozen.sides == 1).sum()),
            recycled=recycled, exact_transitions=True, exact_carry_resets=True,
            scope="Adapter transition/dispatch audit, not an arena strength evaluation",
            frozen_checkpoint_sha256=args.sha256,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        frozen.close()
        reference.close()


if __name__ == "__main__":
    main()

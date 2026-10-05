"""Measure captures from a frozen parent under different sampling temperatures."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.puffer import TrainingRecord, training_lineage_seeds

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


@jax.jit
def sample(key, logits, legal, temperature):
    return jax.random.categorical(key, jnp.where(legal, logits / temperature, -jnp.inf), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1616)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1., .25, .0625, 0.])
    args = parser.parse_args()
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Sampling probe requires GPU")
    if args.games < 16 or args.games % 16 or any(t < 0 or not np.isfinite(t) for t in args.temperatures):
        raise ValueError("Require whole Classic batches and finite nonnegative temperatures")
    policy = SpatialPlayerPolicy(args.bundle)
    lineage = TrainingRecord.model_validate_json((args.bundle / "training.json").read_text())
    if args.seed in training_lineage_seeds(args.bundle, lineage):
        raise ValueError("Probe seed overlaps training lineage")
    record = json.loads((args.bundle / "build.json").read_text())
    options = record["config"]["python_environment"]["options"].copy()
    for k in tuple(options):
        if k.startswith("frozen_"):
            options.pop(k)
    options.update(parallel_games=args.games, coworld_pool_size=args.games,
                   shaping_weight=0., reward_scale=1., land_gain_reward_weight=0.,
                   teacher=None, teacher_rollouts=False, supervise_teacher=False)
    args.output.mkdir(parents=True, exist_ok=False)

    @jax.jit
    def forward(values):
        with jax.default_matmul_precision("highest"):
            return policy._forward(values, jnp)

    results = []
    baseline_hash = None
    for temperature in args.temperatures:
        out = args.output / f"temperature-{temperature:g}"
        out.mkdir()
        env = BatchedGeneralsSelfPlayPufferEnvironment(
            context=EnvironmentContext(seed=args.seed, index=0, mode="train", output=out), **options)
        finished = np.zeros(args.games, bool)
        outcomes = np.zeros((args.games, 2), np.float32)
        try:
            values, masks = env.reset_device(f"{args.seed}:0:0")
            digest = hashlib.sha256()
            for leaf in jax.tree.leaves(env.states):
                digest.update(np.asarray(leaf).tobytes())
            initial_hash = digest.hexdigest()
            if baseline_hash is None:
                baseline_hash = initial_hash
            assert initial_hash == baseline_hash, "Temperature panels must start on identical maps"
            key = jax.random.PRNGKey(args.seed)
            for tick in range(env.horizon):
                predictions = forward(values)
                assert bool(jnp.isfinite(predictions).all())
                legal = masks != 0
                logits = predictions[:, :3529]
                if temperature == 0:
                    actions = jnp.argmax(jnp.where(legal, logits, -jnp.inf), axis=-1)
                else:
                    actions = sample(jax.random.fold_in(key, tick), logits, legal, temperature)
                assert bool(jnp.take_along_axis(legal, actions[:, None], axis=1).all())
                if tick == 0:
                    np.testing.assert_allclose(predictions, policy.forward(np.asarray(values)), rtol=2e-5, atol=2e-5)
                values, masks, rewards, done, _ = env.step_device(actions[:, None])
                reward = np.asarray(rewards).reshape(args.games, 2)
                terminal = np.asarray(done).reshape(args.games, 2)
                assert np.isin(reward, [-1, 0, 1]).all() and (reward.sum(axis=1) == 0).all()
                assert np.array_equal(terminal[:, 0], terminal[:, 1])
                ended = (terminal[:, 0] != 0) & ~finished
                outcomes[ended] = reward[ended]
                finished |= ended
                if finished.all():
                    break
            assert finished.all(), "Every first episode must finish"
            result = dict(temperature=temperature, games=args.games,
                          captures=int((outcomes[:, 0] != 0).sum()),
                          draws=int((outcomes[:, 0] == 0).sum()),
                          seat0_wins=int((outcomes[:, 0] > 0).sum()),
                          seat1_wins=int((outcomes[:, 0] < 0).sum()),
                          initial_state_sha256=initial_hash, ticks=tick + 1)
            np.save(out / "outcomes.npy", outcomes)
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    result = dict(scope="Frozen parent sampling on first Classic self-play episodes; not trained-policy strength",
                  seed=args.seed, checkpoint_sha256=hashlib.sha256((args.bundle / "policy.bin").read_bytes()).hexdigest(),
                  results=results)
    (args.output / "sampling-probe.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

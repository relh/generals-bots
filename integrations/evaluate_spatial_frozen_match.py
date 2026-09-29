"""Count first-episode capture outcomes between two immutable spatial actors."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.puffer import TrainingRecord, training_lineage_seeds

from integrations.evaluate_coworld_frozen_greedy import sample_flat_logits
from integrations.spatial_policy_bundle import SpatialPlayerPolicy
from integrations.spatial_selfplay import SpatialFrozenOpponentPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--opponent-bundle", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="Learner run containing its initialization lineage")
    parser.add_argument("--opponent-run", type=Path, help="Opponent run containing its initialization lineage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1513)
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2 or args.pool_size <= 0:
        raise ValueError("Require a positive even game count and positive pool size")
    if not np.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    if args.sampling_temperature != 1 and args.sample_seed is None:
        raise ValueError("Nondefault temperature requires sampled learner actions")
    for bundle, explicit_run in ((args.bundle, args.run), (args.opponent_bundle, args.opponent_run)):
        training = TrainingRecord.model_validate_json((bundle / "training.json").read_text())
        run = explicit_run or bundle.parent / "run"
        if (run / "training.json").exists():
            if (run / "training.json").read_bytes() != (bundle / "training.json").read_bytes():
                raise ValueError("Training lineage run does not match its policy bundle")
        elif training.config.initialize:
            raise ValueError("Initialized policy requires its source run for lineage verification")
        else:
            run = bundle
        if args.seed in training_lineage_seeds(run, training):
            raise ValueError("Match seed must be absent from both training lineages")
    if not args.smoke_cpu and jax.devices()[0].platform != "gpu":
        raise RuntimeError("Frozen match evaluation requires GPU execution")
    policy = SpatialPlayerPolicy(args.bundle)
    record = json.loads((args.bundle / "build.json").read_text())
    options = record["config"]["python_environment"]["options"].copy()
    for k in ("frozen_bundle", "frozen_build", "frozen_training", "frozen_checkpoint", "frozen_sha256"):
        options.pop(k, None)
    options.update(parallel_games=args.games, coworld_pool_size=args.pool_size,
                   shaping_weight=0.0, reward_scale=1.0, land_gain_reward_weight=0.0)
    if options.get("public_scalar_features"):
        # Each portable actor applies its own ablation; both receive the full
        # public view so a zero-scalar candidate can face a full-scalar actor.
        options["public_scalar_ablation"] = False
    if args.smoke_cpu:
        options.update(require_gpu=False, horizon=4)
    args.output.mkdir(parents=True, exist_ok=False)
    context = EnvironmentContext(seed=args.seed, index=0, mode="train", output=args.output)
    env = SpatialFrozenOpponentPufferEnvironment(frozen_bundle=str(args.opponent_bundle), context=context, **options)

    @jax.jit
    def forward(values):
        with jax.default_matmul_precision("highest"):
            return policy._forward(values, jnp)

    start = time.monotonic()
    finished = np.zeros(args.games, bool)
    outcomes = np.zeros(args.games, np.float32)
    try:
        values, masks = env.reset_device(f"{args.seed}:0:0")
        sides = np.asarray(env.sides)
        assert (sides == 0).sum() == (sides == 1).sum() == args.games // 2
        np.save(args.output / "initial_sides.npy", sides)
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(env.states)]
        assert all(leaf.shape[0] == args.games for leaf in leaves)
        hashes = []
        for row in range(args.games):
            digest = hashlib.sha256()
            for leaf in leaves:
                digest.update(str((leaf.dtype.str, leaf.shape[1:])).encode())
                digest.update(leaf[row].tobytes())
            hashes.append(digest.hexdigest())
        np.save(args.output / "initial_state_sha256.npy", np.asarray(hashes, dtype="U64"))
        for turn in range(env.horizon):
            outputs = np.asarray(forward(values))
            assert outputs.shape == (args.games, 3530) and np.isfinite(outputs).all()
            legal = np.asarray(masks, bool)
            if args.sample_seed is None:
                chosen = np.argmax(np.where(legal, outputs[:, :3529], -np.inf), axis=1)
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn)
                chosen = np.asarray(sample_flat_logits(
                    key, jnp.asarray(outputs[:, :3529]) / args.sampling_temperature, jnp.asarray(legal),
                ))
            actions = chosen.astype(np.int32)[:, None]
            assert legal[np.arange(args.games), actions[:, 0]].all()
            if turn == 0:
                reference = policy.forward(np.asarray(values))
                assert np.allclose(outputs, reference, rtol=2e-5, atol=2e-5)
                if args.sample_seed is None:
                    assert np.array_equal(actions[:, 0], np.argmax(np.where(legal, reference[:, :3529], -np.inf), axis=1))
            values, masks, rewards, done, _ = env.step_device(jnp.asarray(actions))
            ended = np.asarray(done, bool) & ~finished
            reward = np.asarray(rewards)
            assert np.isfinite(reward).all() and np.isin(reward[ended], (-1, 0, 1)).all()
            outcomes[ended] = reward[ended]
            finished |= ended
            if turn % 100 == 0:
                print(json.dumps(dict(turn=turn + 1, finished=int(finished.sum()))), flush=True)
            if finished.all():
                break
        assert finished.all(), "Every first episode must reach capture or truncation"
    finally:
        env.close()
    result = dict(scope="First held-out episodes between frozen public-view actors; CPU smoke is not strength evidence",
                  smoke_cpu=args.smoke_cpu, games=args.games, seed=args.seed, pool_size=args.pool_size,
                  held_out=True, unique_initial_states=len(set(hashes)),
                  action_selection="argmax" if args.sample_seed is None else "sample",
                  sample_seed=args.sample_seed,
                  sampling_temperature=args.sampling_temperature if args.sample_seed is not None else None,
                  opponent_action_selection="argmax",
                  checkpoint_sha256=hashlib.sha256((args.bundle / "policy.bin").read_bytes()).hexdigest(),
                  opponent_sha256=hashlib.sha256((args.opponent_bundle / "policy.bin").read_bytes()).hexdigest(),
                  episode_limit=env.horizon,
                  coworld_classic_rules=env.base.env.coworld_classic_rules,
                  pool_generation=int(env._pool_generation),
                  wins=int((outcomes > 0).sum()), losses=int((outcomes < 0).sum()), draws=int((outcomes == 0).sum()),
                  score=float(outcomes.mean()), turns=turn + 1, wall_seconds=time.monotonic() - start)
    np.save(args.output / "outcomes.npy", outcomes)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

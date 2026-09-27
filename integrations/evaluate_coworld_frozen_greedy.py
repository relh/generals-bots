"""Evaluate the hosted argmax policy on verified, held-out Classic games."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig
from metta_training.puffer import TrainingRecord, training_lineage_seeds

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true", help="Pinned default Puffer5 MinGRU checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--training-pool-episode", type=int,
                        help="Diagnostic only: evaluate the latest pool seen by this checkpoint")
    parser.add_argument("--sample-seed", type=int, help="Native sampling diagnostic instead of hosted argmax")
    parser.add_argument("--opponent", choices=("expander_harvester", "sentinel", "strong_mixed"), required=True)
    args = parser.parse_args()
    assert args.games > 0 and jax.devices()[0].platform == "gpu"
    assert args.sample_seed is None or args.native
    training = TrainingRecord.model_validate_json((args.run / "training.json").read_text())
    if args.training_pool_episode is None:
        assert args.seed not in training_lineage_seeds(args.run, training)
    else:
        assert args.native and args.training_pool_episode >= 0
        assert args.seed == training.config.seed
    manifest = json.loads(args.build.read_text())
    assert manifest["model_sha256"] == training.build.model_sha256
    if args.native:
        policy = NativePufferPolicy(args.build, args.run / "training.json", args.checkpoint, args.sha256)
        recurrent_state = policy.initial_state(args.games)
    else:
        policy = FrozenPolicy(FrozenPolicyConfig(
            build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0",
        ))
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["coworld_classic"] and not options["teacher_rollouts"]
    options.update(parallel_games=args.games, opponent=args.opponent, supervise_teacher=False)
    if args.pool_size is not None:
        if args.training_pool_episode is not None:
            assert args.pool_size == options["coworld_pool_size"]
        options["coworld_pool_size"] = args.pool_size
    args.output.mkdir(parents=True, exist_ok=False)
    context = EnvironmentContext(seed=args.seed, index=0, mode="evaluate", output=args.output)
    env = BatchedGeneralsPufferEnvironment(context=context, **options)
    episode = 0 if args.training_pool_episode is None else args.training_pool_episode
    reset_seed = f"{args.seed}:0:{episode}"
    if args.training_pool_episode is not None:
        agents = manifest["config"]["python_environment"]["spec"]["agents"]
        assert int(args.checkpoint.stem) // (agents * env.horizon) == episode
    if not args.native:
        policy.reset(reset_seed)
    seats = list(range(args.games))
    start = time.monotonic()
    try:
        observation = env.reset(reset_seed)
        initial_leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(env.states)]
        assert all(leaf.shape[0] == args.games for leaf in initial_leaves)
        initial_hashes = []
        for lane in range(args.games):
            digest = hashlib.sha256()
            for leaf in initial_leaves:
                digest.update(str((leaf.dtype.str, leaf.shape[1:])).encode())
                digest.update(leaf[lane].tobytes())
            initial_hashes.append(digest.hexdigest())
        np.save(args.output / "initial_state_sha256.npy", np.asarray(initial_hashes, dtype="U64"))
        np.save(args.output / "initial_sides.npy", np.asarray(env.sides))
        np.save(args.output / "initial_opponent_ids.npy", np.asarray(env.opponent_ids))
        for turn in range(env.horizon):
            if args.native:
                actions, recurrent_state = policy.actions(
                    np.asarray(observation.values), np.asarray(observation.action_masks), recurrent_state,
                    key=(None if args.sample_seed is None else
                         jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn)),
                )
                actions = np.asarray(actions, dtype=np.int32)
            else:
                predictions = policy.predict_many(seats, observation)
                actions = np.asarray([
                    (np.argmax(prediction.probabilities[:1765]), np.argmax(prediction.probabilities[1765:]))
                    for prediction in predictions
                ], dtype=np.int32)
            masks = np.asarray(observation.action_masks, dtype=bool)
            assert masks[np.arange(args.games), actions[:, 0]].all()
            assert masks[np.arange(args.games), 1765 + actions[:, 1]].all()
            transition = env.step(actions)
            observation = transition.observation
            if turn % 50 == 0:
                print(json.dumps(dict(turn=turn + 1, finished=int(env.finished.sum()))), flush=True)
            if transition.episode_done:
                break
        assert env.finished.all()
        outcomes = env.outcomes.copy()
    finally:
        env.close()
    result = dict(
        scope=("Training-pool diagnostic; does not establish held-out or hosted performance"
               if args.training_pool_episode is not None else
               "Frozen GPU sampling diagnostic; hosted player currently uses argmax"
               if args.sample_seed is not None else
               "Frozen GPU argmax inference on Classic maps; hosted service startup remains separate"),
        held_out=args.training_pool_episode is None, training_pool_episode=args.training_pool_episode,
        action_selection="argmax_per_head" if args.sample_seed is None else "sample_per_head",
        sample_seed=args.sample_seed, seed=args.seed, reset_seed=reset_seed,
        games=args.games, opponent=args.opponent, checkpoint_sha256=args.sha256,
        unique_initial_states=len(set(initial_hashes)),
        sampling="Pool samples; initial state hashes, sides and opponent IDs saved",
        model_sha256=manifest["model_sha256"], native_mingru=args.native, options=options,
        wins=int((outcomes > 0).sum()), losses=int((outcomes < 0).sum()), draws=int((outcomes == 0).sum()),
        score=float(outcomes.mean()), perf=float((outcomes.mean() + 1) / 2),
        turns=turn + 1, wall_seconds=time.monotonic() - start,
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(args.output / "outcomes.npy", outcomes)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

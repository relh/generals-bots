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
from integrations.puffer_codec import hinted_replay_indices


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
    parser.add_argument(
        "--training-pool-episode", type=int, help="Diagnostic only: evaluate the latest pool seen by this checkpoint"
    )
    parser.add_argument("--sample-seed", type=int, help="Native sampling diagnostic instead of hosted argmax")
    parser.add_argument(
        "--reward-diagnostics",
        action="store_true",
        help="Count raw rewards affected by the native learner's [-1, 1] clamp",
    )
    parser.add_argument("--force-hint-move", action="store_true", help="Diagnostic: replace the native move head")
    parser.add_argument("--force-hint-split", action="store_true", help="Diagnostic: replace the native split head")
    parser.add_argument(
        "--opponent", choices=("random", "expander_harvester", "sentinel", "strong_mixed"), required=True
    )
    args = parser.parse_args()
    assert args.games > 0 and jax.devices()[0].platform == "gpu"
    assert args.sample_seed is None or args.native
    assert not (args.force_hint_move or args.force_hint_split) or args.native and args.sample_seed is None
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
        policy = FrozenPolicy(
            FrozenPolicyConfig(
                build=args.build,
                checkpoint=args.checkpoint,
                sha256=args.sha256,
                device="cuda:0",
            )
        )
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["coworld_classic"] and not options["teacher_rollouts"]
    if args.force_hint_move or args.force_hint_split:
        assert options["prior_hint_features"] and options["expander_hint_features"] and options["context_hint_features"]
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
    reward_diagnostics = dict(
        active_steps=0,
        clipped_steps=0,
        clipped_terminal_steps=0,
        min_raw_reward=float("inf"),
        max_raw_reward=float("-inf"),
        total_absolute_clamp_change=0.0,
    )
    intervention = dict(active_steps=0, changed_moves=0, changed_splits=0)
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
                    np.asarray(observation.values),
                    np.asarray(observation.action_masks),
                    recurrent_state,
                    **(
                        {}
                        if args.sample_seed is None
                        else dict(key=jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn))
                    ),
                )
                actions = np.asarray(actions, dtype=np.int32).copy()
            else:
                predictions = policy.predict_many(seats, observation)
                actions = np.asarray(
                    [
                        (np.argmax(prediction.probabilities[:1765]), np.argmax(prediction.probabilities[1765:]))
                        for prediction in predictions
                    ],
                    dtype=np.int32,
                )
            masks = np.asarray(observation.action_masks, dtype=bool)
            if args.force_hint_move or args.force_hint_split:
                hint = hinted_replay_indices(observation.values, 21, channels=14)
                active = ~env.finished
                assert masks[np.arange(args.games)[active], hint[active, 0]].all()
                intervention["active_steps"] += int(active.sum())
                intervention["changed_moves"] += int((active & (actions[:, 0] != hint[:, 0])).sum())
                intervention["changed_splits"] += int(
                    (
                        active & (actions[:, 0] == hint[:, 0]) & (actions[:, 0] != 1764) & (actions[:, 1] != hint[:, 1])
                    ).sum()
                )
                if args.force_hint_move:
                    actions[active, 0] = hint[active, 0]
                if args.force_hint_split:
                    actions[active, 1] = np.where(hint[active, 0] == 1764, 0, hint[active, 1])
            assert masks[np.arange(args.games), actions[:, 0]].all()
            assert masks[np.arange(args.games), 1765 + actions[:, 1]].all()
            active = ~env.finished.copy() if args.reward_diagnostics else None
            transition = env.step(actions)
            if args.reward_diagnostics:
                rewards = np.asarray(transition.rewards, dtype=np.float32)[active]
                terminals = np.asarray(transition.terminated, dtype=bool)[active]
                assert np.isfinite(rewards).all()
                clipped = np.abs(rewards) > 1.0
                reward_diagnostics["active_steps"] += int(rewards.size)
                reward_diagnostics["clipped_steps"] += int(clipped.sum())
                reward_diagnostics["clipped_terminal_steps"] += int((clipped & terminals).sum())
                reward_diagnostics["min_raw_reward"] = min(reward_diagnostics["min_raw_reward"], float(rewards.min()))
                reward_diagnostics["max_raw_reward"] = max(reward_diagnostics["max_raw_reward"], float(rewards.max()))
                reward_diagnostics["total_absolute_clamp_change"] += float(
                    np.abs(rewards - np.clip(rewards, -1.0, 1.0)).sum(dtype=np.float64)
                )
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
        scope=(
            "Frozen action intervention diagnostic; not the hosted policy"
            if args.force_hint_move or args.force_hint_split
            else "Training-pool diagnostic; does not establish held-out or hosted performance"
            if args.training_pool_episode is not None
            else "Frozen GPU sampling diagnostic; hosted player currently uses argmax"
            if args.sample_seed is not None
            else "Frozen GPU argmax inference on Classic maps; hosted service startup remains separate"
        ),
        held_out=args.training_pool_episode is None,
        training_pool_episode=args.training_pool_episode,
        action_selection=(
            "argmax_with_hint_intervention"
            if args.force_hint_move or args.force_hint_split
            else "argmax_per_head"
            if args.sample_seed is None
            else "sample_per_head"
        ),
        sample_seed=args.sample_seed,
        seed=args.seed,
        reset_seed=reset_seed,
        games=args.games,
        opponent=args.opponent,
        checkpoint_sha256=args.sha256,
        unique_initial_states=len(set(initial_hashes)),
        sampling="Pool samples; initial state hashes, sides and opponent IDs saved",
        model_sha256=manifest["model_sha256"],
        native_mingru=args.native,
        options=options,
        force_hint_move=args.force_hint_move,
        force_hint_split=args.force_hint_split,
        intervention=intervention if args.force_hint_move or args.force_hint_split else None,
        wins=int((outcomes > 0).sum()),
        losses=int((outcomes < 0).sum()),
        draws=int((outcomes == 0).sum()),
        score=float(outcomes.mean()),
        perf=float((outcomes.mean() + 1) / 2),
        turns=turn + 1,
        wall_seconds=time.monotonic() - start,
    )
    if args.reward_diagnostics:
        result["reward_diagnostics"] = dict(
            scope="Frozen argmax trajectories; raw environment rewards versus native learner clamp",
            **reward_diagnostics,
        )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(args.output / "outcomes.npy", outcomes)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

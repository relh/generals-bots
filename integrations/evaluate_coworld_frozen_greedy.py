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


def greedy_declared_heads(predictions, sizes):
    probabilities = np.asarray([p.probabilities for p in predictions], dtype=np.float32)
    if probabilities.shape != (len(predictions), sum(sizes)) or not np.isfinite(probabilities).all():
        raise ValueError("Frozen probabilities differ from the declared action heads")
    boundaries = np.cumsum((0, *sizes))
    return np.stack([
        np.argmax(probabilities[:, start:stop], axis=1)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ], axis=1).astype(np.int32)


@jax.jit
def sample_flat_logits(key, logits, legal):
    import jax.numpy as jnp

    return jax.random.categorical(key, jnp.where(legal, logits, -jnp.inf), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true", help="Pinned default Puffer5 MinGRU checkpoint")
    parser.add_argument("--spatial-bundle", type=Path, help="Verified portable spatial weights with GPU inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument(
        "--training-pool-episode", type=int, help="Diagnostic only: evaluate the latest pool seen by this checkpoint"
    )
    parser.add_argument("--action-diagnostics", action="store_true", help="Record public directional-codec move behavior by game phase")
    parser.add_argument("--sample-seed", type=int, help="Native or spatial sampling diagnostic instead of hosted argmax")
    parser.add_argument("--sampling-temperature", type=float, default=1.0,
                        help="Spatial categorical sampling temperature; match the training runtime")
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
    assert not (args.native and args.spatial_bundle)
    assert args.games > 0 and jax.devices()[0].platform == "gpu"
    assert args.sample_seed is None or args.native or args.spatial_bundle
    if not np.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    if args.sampling_temperature != 1 and not (args.spatial_bundle and args.sample_seed is not None):
        raise ValueError("Nondefault sampling temperature requires a sampled spatial policy")
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
        from integrations.native_puffer_policy import NativePufferPolicy

        policy = NativePufferPolicy(args.build, args.run / "training.json", args.checkpoint, args.sha256)
        recurrent_state = policy.initial_state(args.games)
    elif args.spatial_bundle:
        import jax.numpy as jnp
        from types import SimpleNamespace
        from integrations.spatial_policy_bundle import SpatialPlayerPolicy

        policy = SpatialPlayerPolicy(args.spatial_bundle)
        assert hashlib.sha256((args.spatial_bundle / "policy.bin").read_bytes()).hexdigest() == args.sha256
        assert json.loads((args.spatial_bundle / "build.json").read_text()) == manifest
        policy.spec = SimpleNamespace(action_sizes=manifest["config"]["fabric"]["action_sizes"])
        @jax.jit
        def spatial_forward(obs):
            with jax.default_matmul_precision("highest"):
                return policy._forward(obs, jnp)
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
    for key in ("frozen_build", "frozen_training", "frozen_checkpoint", "frozen_sha256", "frozen_bundle"):
        options.pop(key, None)
    assert options["coworld_classic"] and not options["teacher_rollouts"]
    if args.force_hint_move or args.force_hint_split:
        from integrations.puffer_codec import hinted_replay_indices

        assert policy.action_sizes == (1765, 2)
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
    action_stats = {
        phase: dict(turns=0, passes=0, moves=0, split_moves=0,
                    source_army_sum=0.0, max_legal_source_army_sum=0.0,
                    source_to_max_ratio_sum=0.0, attacks_visible_enemy=0,
                    attacks_visible_enemy_winnable=0, attacks_visible_enemy_unwinnable=0,
                    moves_into_owned=0, moves_into_neutral=0, moves_into_fog=0,
                    own_land_sum=0.0, enemy_land_sum=0.0,
                    own_army_sum=0.0, enemy_army_sum=0.0)
        for phase in ("early_0_99", "middle_100_199", "late_200_plus")
    } if args.action_diagnostics else None
    if args.action_diagnostics:
        assert options["directional_features"] and env.spec.observation_size in (4851, 5292, 7056)
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
            elif args.spatial_bundle:
                values = np.asarray(observation.values, np.float32)
                outputs = np.asarray(spatial_forward(jnp.asarray(values)))
                assert outputs.shape == (args.games, 3530) and np.isfinite(outputs).all()
                if turn == 0:
                    reference = policy.forward(values)
                    assert np.allclose(outputs, reference, rtol=2e-5, atol=2e-5)
                    legal = np.asarray(observation.action_masks, bool)
                    assert np.array_equal(np.argmax(np.where(legal, outputs[:, :3529], -np.inf), axis=1),
                                          np.argmax(np.where(legal, reference[:, :3529], -np.inf), axis=1))
                legal = np.asarray(observation.action_masks, bool)
                if args.sample_seed is None:
                    actions = np.argmax(np.where(legal, outputs[:, :3529], -np.inf), axis=1)
                else:
                    key = jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn)
                    actions = np.asarray(sample_flat_logits(
                        key, jnp.asarray(outputs[:, :3529]) / args.sampling_temperature, jnp.asarray(legal),
                    ))
                actions = actions.astype(np.int32)[:, None]
            else:
                predictions = policy.predict_many(seats, observation)
                actions = greedy_declared_heads(predictions, env.spec.action_sizes)
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
            offset = 0
            sizes = tuple(env.spec.action_sizes)
            assert tuple(policy.action_sizes if args.native else policy.spec.action_sizes) == sizes
            assert actions.shape == (args.games, len(sizes))
            for head, size in enumerate(sizes):
                assert masks[np.arange(args.games), offset + actions[:, head]].all()
                offset += size
            if action_stats is not None:
                phase = "early_0_99" if turn < 100 else "middle_100_199" if turn < 200 else "late_200_plus"
                stats = action_stats[phase]
                active_rows = np.flatnonzero(~env.finished)
                chosen = actions[active_rows, 0]
                flat_actions = len(env.spec.action_sizes) == 1
                move_logits = (8 if flat_actions else 4) * 21 * 21
                moving = chosen < move_logits
                stats["turns"] += len(active_rows)
                stats["passes"] += int((~moving).sum())
                stats["moves"] += int(moving.sum())
                values = np.asarray(observation.values, dtype=np.float32).reshape(args.games, -1, 21 * 21)
                armies = np.rint(np.expm1(values[:, 0] * 8.0))
                own = values[active_rows, 4] > 0
                enemy = values[active_rows, 5] > 0
                stats["own_land_sum"] += float(own.sum())
                stats["enemy_land_sum"] += float(enemy.sum())
                stats["own_army_sum"] += float((armies[active_rows] * own).sum())
                stats["enemy_army_sum"] += float((armies[active_rows] * enemy).sum())
                if moving.any():
                    rows = active_rows[moving]
                    choices = chosen[moving]
                    cells = 21 * 21
                    source_cells = choices % cells
                    directions = (choices // cells) % 4
                    selected_armies = armies[rows, source_cells]
                    legal_sources = masks[:, :move_logits].reshape(
                        args.games, 8 if flat_actions else 4, cells
                    ).any(axis=1)
                    max_armies = np.where(legal_sources[rows], armies[rows], 0).max(axis=1)
                    assert (max_armies >= selected_armies).all() and (selected_armies > 1).all()
                    stats["source_army_sum"] += float(selected_armies.sum())
                    stats["max_legal_source_army_sum"] += float(max_armies.sum())
                    stats["source_to_max_ratio_sum"] += float((selected_armies / max_armies).sum())
                    stats["split_moves"] += int((choices // (4 * cells)).sum()) if flat_actions else int(
                        (actions[rows, 1] == 1).sum()
                    )
                    source_r, source_c = divmod(source_cells, 21)
                    dest_r = source_r + np.asarray((-1, 1, 0, 0))[directions]
                    dest_c = source_c + np.asarray((0, 0, -1, 1))[directions]
                    assert ((dest_r >= 0) & (dest_r < 21) & (dest_c >= 0) & (dest_c < 21)).all()
                    dest_cells = dest_r * 21 + dest_c
                    attacking_enemy = values[rows, 5, dest_cells] > 0
                    moving_armies = np.where(
                        (choices // (4 * cells) == 1) if flat_actions else (actions[rows, 1] == 1),
                        selected_armies // 2, selected_armies - 1,
                    )
                    winnable = moving_armies > armies[rows, dest_cells]
                    stats["attacks_visible_enemy"] += int(attacking_enemy.sum())
                    stats["attacks_visible_enemy_winnable"] += int((attacking_enemy & winnable).sum())
                    stats["attacks_visible_enemy_unwinnable"] += int((attacking_enemy & ~winnable).sum())
                    stats["moves_into_owned"] += int((values[rows, 4, dest_cells] > 0).sum())
                    stats["moves_into_fog"] += int((values[rows, 6, dest_cells] > 0).sum())
                    stats["moves_into_neutral"] += int((
                        (values[rows, 4, dest_cells] == 0)
                        & (values[rows, 5, dest_cells] == 0)
                        & (values[rows, 6, dest_cells] == 0)
                    ).sum())
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
        sampling_temperature=args.sampling_temperature if args.sample_seed is not None else None,
        seed=args.seed,
        reset_seed=reset_seed,
        games=args.games,
        opponent=args.opponent,
        checkpoint_sha256=args.sha256,
        unique_initial_states=len(set(initial_hashes)),
        sampling="Pool samples; initial state hashes, sides and opponent IDs saved",
        model_sha256=manifest["model_sha256"],
        native_mingru=args.native,
        spatial_bundle_gpu=bool(args.spatial_bundle),
        options=options,
        coworld_classic_rules=env.base.env.coworld_classic_rules,
        episode_limit=env.horizon,
        force_hint_move=args.force_hint_move,
        force_hint_split=args.force_hint_split,
        intervention=intervention if args.force_hint_move or args.force_hint_split else None,
        action_stats=action_stats,
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

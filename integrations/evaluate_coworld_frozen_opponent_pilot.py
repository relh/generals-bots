"""Evaluate a legacy Fabric checkpoint on held-out Classic games."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.puffer_codec import hinted_replay_indices


def legacy_actions_many(
    policy: FrozenPolicy, seats: list[int], observation, *, return_logits: bool = False,
    counterfactual_hint_scale: float = 1.0,
):
    """Batch the pinned legacy Fabric forward pass used by the training build."""
    values = np.asarray(observation.values, dtype=np.float32)
    if counterfactual_hint_scale != 1.0:
        values = values.copy().reshape((-1, 14, 21, 21))
        values[:, 2:8] *= counterfactual_hint_scale
        values = values.reshape((values.shape[0], -1))
    masks = np.asarray(observation.action_masks, dtype=bool)
    if values.shape != (len(seats), policy.policy.observation_size):
        raise ValueError("Frozen observation dimensions differ from the policy")
    if masks.shape != (len(seats), sum(policy.policy.action_sizes)):
        raise ValueError("Frozen action-mask dimensions differ from the policy")
    transported = np.zeros((len(seats), policy.policy.input_size), np.float32)
    transported[:, : values.shape[1]] = values
    zero_state = bytes(policy.policy.state_words * 4)
    state_in = b"".join(policy.states.setdefault(seat, zero_state) for seat in seats)
    with jax.default_device(policy.device):
        outputs, states, _ = policy.policy.forward(
            policy.parameters, state_in, transported.tobytes(), bytes(4 * len(seats)), len(seats), 1, True
        )
    logits = np.frombuffer(outputs, np.float32).reshape(len(seats), policy.policy.output_size)[:, :-1]
    state_rows = np.frombuffer(states, np.float32).reshape(len(seats), policy.policy.state_words)
    for seat, state in zip(seats, state_rows, strict=True):
        policy.states[seat] = state.tobytes()
    heads = []
    offset = 0
    for size in policy.policy.action_sizes:
        legal = masks[:, offset : offset + size]
        if not legal.any(axis=1).all():
            raise ValueError("Every frozen-policy head requires a legal action")
        heads.append(np.argmax(np.where(legal, logits[:, offset : offset + size], -np.inf), axis=1))
        offset += size
    actions = np.stack(heads, axis=1).astype(np.int32)
    return (actions, logits) if return_logits else actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true", help="Pinned default Puffer5 MinGRU checkpoint")
    parser.add_argument("--diagnostic-checkpoint", action="store_true", help="Evaluate an explicitly altered checkpoint")
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
    parser.add_argument("--force-hint-move", action="store_true", help="Diagnostic: replace the move head")
    parser.add_argument("--force-hint-split", action="store_true", help="Diagnostic: replace the split head")
    parser.add_argument("--force-full-split", action="store_true", help="Diagnostic: use full-army moves")
    parser.add_argument("--own-destination-logit-penalty", type=float, default=0.0,
                        help="Diagnostic: subtract this logit from moves into a public owned tile")
    parser.add_argument("--hint-audit", action="store_true", help="Measure frozen action agreement and probability on hint-driven states")
    parser.add_argument("--teacher-action-audit", action="store_true",
                        help="Measure agreement with scripted actions from the current public game state")
    parser.add_argument("--audit-turns", type=int, help="Stop a hint audit after this many turns")
    parser.add_argument("--counterfactual-hint-scale", type=float, default=1.0,
                        help="Hint audit only: scale public hint planes before frozen policy forward")
    parser.add_argument(
        "--opponent", choices=("random", "expander_harvester", "sentinel", "strong_mixed"), required=True
    )
    args = parser.parse_args()
    assert args.games > 0 and jax.devices()[0].platform == "gpu"
    assert args.own_destination_logit_penalty >= 0
    assert args.sample_seed is None or args.native
    assert not (args.force_hint_move or args.force_hint_split) or args.sample_seed is None
    assert not (args.force_full_split and args.force_hint_split)
    assert not args.native and args.training_pool_episode is None
    assert args.audit_turns is None or ((args.hint_audit or args.teacher_action_audit) and args.audit_turns > 0)
    assert not (args.hint_audit and args.teacher_action_audit)
    assert not args.hint_audit or (args.force_hint_move and args.force_hint_split)
    assert args.counterfactual_hint_scale == 1.0 or (
        args.hint_audit and 0.0 <= args.counterfactual_hint_scale < 1.0
    )
    record = json.loads((args.run / "training.json").read_text())
    completed_path = args.run / "completed.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else None
    assert args.seed != record["config"]["seed"]
    if not args.diagnostic_checkpoint:
        assert completed is not None, "Completed run metadata is required outside diagnostic mode"
        assert args.checkpoint.relative_to(args.run).as_posix() in completed["checkpoints"]
    manifest = json.loads(args.build.read_text())
    assert manifest == record["build"]
    policy = FrozenPolicy(
        FrozenPolicyConfig(
            build=args.build,
            checkpoint=args.checkpoint,
            sha256=args.sha256,
            device="cuda:0",
        )
        )
    options = manifest["config"]["python_environment"]["options"].copy()
    for frozen_asset in ("frozen_build", "frozen_checkpoint", "frozen_sha256", "scripted_hint_fraction"):
        options.pop(frozen_asset, None)
    assert options["coworld_classic"] and not options["teacher_rollouts"]
    if args.force_hint_move or args.force_hint_split:
        assert options["prior_hint_features"] and options["expander_hint_features"] and options["context_hint_features"]
    options.update(parallel_games=args.games, opponent=args.opponent,
                   supervise_teacher=args.teacher_action_audit, deduplicate_opponent_branches=False)
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
    hint_audit = dict(active=0, move_match=0, split_labeled=0, split_match=0, joint_match=0,
                      teacher_pass=0, student_pass=0, both_pass=0,
                      teacher_move=0, student_move_when_teacher_moves=0, move_match_when_teacher_moves=0,
                      move_probability_sum=0.0, split_probability_sum=0.0,
                      move_negative_log_probability_sum=0.0, split_negative_log_probability_sum=0.0)
    teacher_action_audit = dict(active=0, move_match=0, teacher_move=0,
                                student_move_when_teacher_moves=0, teacher_pass=0,
                                student_pass=0, move_probability_sum=0.0,
                                move_negative_log_probability_sum=0.0)
    action_stats = {
        phase: dict(turns=0, passes=0, moves=0, split_moves=0,
                    source_army_sum=0.0, max_legal_source_army_sum=0.0,
                    source_to_max_ratio_sum=0.0, attacks_visible_enemy=0,
                    attacks_visible_enemy_winnable=0, attacks_visible_enemy_unwinnable=0,
                    moves_into_owned=0, moves_into_neutral=0, moves_into_fog=0,
                    own_land_sum=0.0, enemy_land_sum=0.0,
                    own_army_sum=0.0, enemy_army_sum=0.0)
        for phase in ("early_0_99", "middle_100_199", "late_200_plus")
    } if manifest["config"]["python_environment"]["spec"]["observation_size"] == 11 * 21 * 21 else None
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
        for turn in range(min(env.horizon, args.audit_turns or env.horizon)):
            forward = legacy_actions_many(
                policy, seats, observation,
                return_logits=args.hint_audit or args.teacher_action_audit
                or args.own_destination_logit_penalty > 0,
                counterfactual_hint_scale=args.counterfactual_hint_scale,
            )
            actions, logits = forward if args.hint_audit or args.teacher_action_audit \
                or args.own_destination_logit_penalty > 0 else (forward, None)
            masks = np.asarray(observation.action_masks, dtype=bool)
            if args.own_destination_logit_penalty:
                if len(env.spec.action_sizes) != 1 or env.spec.action_sizes[0] != 8 * 21 * 21 + 1:
                    raise ValueError("The destination intervention requires flat 21x21 actions")
                cells = 21 * 21
                source_cells = np.arange(cells)
                source_r, source_c = divmod(source_cells, 21)
                dest_r = np.clip(source_r[None, :] + np.asarray((-1, 1, 0, 0))[:, None], 0, 20)
                dest_c = np.clip(source_c[None, :] + np.asarray((0, 0, -1, 1))[:, None], 0, 20)
                destination = (dest_r * 21 + dest_c).reshape(-1)
                owned = np.asarray(observation.values, np.float32).reshape(args.games, 11, cells)[:, 4]
                owned_destination = np.tile(owned[:, destination], (1, 2))
                adjusted = logits[:, :8 * cells + 1].copy()
                adjusted[:, :8 * cells] -= args.own_destination_logit_penalty * owned_destination
                proposed = np.argmax(np.where(masks[:, :8 * cells + 1], adjusted, -np.inf), axis=1)
                active = ~env.finished
                intervention["active_steps"] += int(active.sum())
                intervention["changed_moves"] += int((active & (actions[:, 0] != proposed)).sum())
                actions[active, 0] = proposed[active]
            if args.teacher_action_audit:
                keys = jnp.broadcast_to(jax.random.PRNGKey(0), (args.games, 2))
                raw_teacher = np.asarray(env._teacher_actions(env.states, env.sides, keys))
                cells = env.base.size**2
                target = np.where(raw_teacher[:, 0] == 1, 4 * cells,
                                  raw_teacher[:, 3] * cells + raw_teacher[:, 1] * env.base.size + raw_teacher[:, 2])
                active = ~env.finished
                rows = np.arange(args.games)[active]
                assert masks[rows, target[active]].all()
                teacher_action_audit["active"] += len(rows)
                teacher_action_audit["move_match"] += int((actions[active, 0] == target[active]).sum())
                teacher_action_audit["teacher_move"] += int((target[active] != 4 * cells).sum())
                teacher_action_audit["student_move_when_teacher_moves"] += int(
                    ((target[active] != 4 * cells) & (actions[active, 0] != 4 * cells)).sum()
                )
                teacher_action_audit["teacher_pass"] += int((target[active] == 4 * cells).sum())
                teacher_action_audit["student_pass"] += int((actions[active, 0] == 4 * cells).sum())
                legal_logits = np.where(masks[active, :1765], logits[active, :1765], -np.inf)
                chosen = legal_logits[np.arange(len(rows)), target[active]]
                maximum = legal_logits.max(axis=1)
                log_probability = chosen - maximum - np.log(np.exp(legal_logits - maximum[:, None]).sum(axis=1))
                teacher_action_audit["move_probability_sum"] += float(np.exp(log_probability).sum())
                teacher_action_audit["move_negative_log_probability_sum"] -= float(log_probability.sum())
            if args.hint_audit:
                hint = hinted_replay_indices(observation.values, 21, channels=14)
                active = ~env.finished
                rows = np.arange(args.games)[active]
                teacher_pass = hint[active, 0] == 1764
                teacher_split = np.where(teacher_pass, 0, hint[active, 1])
                assert masks[rows, hint[active, 0]].all()
                assert masks[rows, 1765 + teacher_split].all()
                hint_audit["active"] += len(rows)
                move_match = actions[active, 0] == hint[active, 0]
                split_match = actions[active, 1] == teacher_split
                hint_audit["move_match"] += int(move_match.sum())
                hint_audit["split_labeled"] += int((~teacher_pass).sum())
                hint_audit["split_match"] += int((~teacher_pass & split_match).sum())
                hint_audit["joint_match"] += int((move_match & (teacher_pass | split_match)).sum())
                student_pass = actions[active, 0] == 1764
                hint_audit["teacher_pass"] += int(teacher_pass.sum())
                hint_audit["student_pass"] += int(student_pass.sum())
                hint_audit["both_pass"] += int((teacher_pass & student_pass).sum())
                hint_audit["teacher_move"] += int((~teacher_pass).sum())
                hint_audit["student_move_when_teacher_moves"] += int((~teacher_pass & ~student_pass).sum())
                hint_audit["move_match_when_teacher_moves"] += int((~teacher_pass & (actions[active, 0] == hint[active, 0])).sum())
                for head, (offset, size) in enumerate(((0, 1765), (1765, 2))):
                    labeled = np.ones(len(rows), dtype=bool) if head == 0 else ~teacher_pass
                    legal_logits = np.where(masks[active, offset:offset + size][labeled],
                                            logits[active, offset:offset + size][labeled], -np.inf)
                    chosen = legal_logits[np.arange(int(labeled.sum())), hint[active, head][labeled]]
                    maximum = legal_logits.max(axis=1)
                    log_probability = chosen - maximum - np.log(np.exp(legal_logits - maximum[:, None]).sum(axis=1))
                    key = "move" if head == 0 else "split"
                    hint_audit[f"{key}_probability_sum"] += float(np.exp(log_probability).sum())
                    hint_audit[f"{key}_negative_log_probability_sum"] -= float(log_probability.sum())
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
            if args.force_full_split:
                flat = len(env.spec.action_sizes) == 1
                changed = (
                    (~env.finished) & (actions[:, 0] >= 4 * 21 * 21) & (actions[:, 0] < 8 * 21 * 21)
                    if flat else (~env.finished) & (actions[:, 0] != 1764) & (actions[:, 1] != 0)
                )
                intervention["active_steps"] += int((~env.finished).sum())
                intervention["changed_splits"] += int(changed.sum())
                if flat:
                    actions[changed, 0] -= 4 * 21 * 21
                else:
                    actions[changed, 1] = 0
            assert masks[np.arange(args.games), actions[:, 0]].all()
            if len(env.spec.action_sizes) == 2:
                assert masks[np.arange(args.games), 1765 + actions[:, 1]].all()
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
                values = np.asarray(observation.values, dtype=np.float32).reshape(args.games, 11, 21 * 21)
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
        episode_complete = bool(env.finished.all())
        assert episode_complete or (args.hint_audit or args.teacher_action_audit) and args.audit_turns is not None
        outcomes = env.outcomes.copy()
    finally:
        env.close()
    result = dict(
        scope=(
            "Frozen action intervention diagnostic; not the hosted policy"
            if args.force_hint_move or args.force_hint_split or args.force_full_split
            or args.own_destination_logit_penalty
            else "Explicitly altered checkpoint diagnostic; not the trained or hosted policy"
            if args.diagnostic_checkpoint
            else "Training-pool diagnostic; does not establish held-out or hosted performance"
            if args.training_pool_episode is not None
            else "Frozen GPU sampling diagnostic; hosted player currently uses argmax"
            if args.sample_seed is not None
            else "Frozen GPU argmax inference on Classic maps; hosted service startup remains separate"
        ),
        held_out=args.training_pool_episode is None,
        training_pool_episode=args.training_pool_episode,
        action_selection=(
            "argmax_with_action_intervention"
            if args.force_hint_move or args.force_hint_split or args.force_full_split
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
        force_full_split=args.force_full_split,
        own_destination_logit_penalty=args.own_destination_logit_penalty,
        intervention=intervention if args.force_hint_move or args.force_hint_split or args.force_full_split else None,
        hint_audit=hint_audit if args.hint_audit else None,
        teacher_action_audit=teacher_action_audit if args.teacher_action_audit else None,
        action_stats=action_stats,
        audit_turns=args.audit_turns,
        counterfactual_hint_scale=args.counterfactual_hint_scale,
        episode_complete=episode_complete,
        wins=int((outcomes > 0).sum()) if episode_complete else None,
        losses=int((outcomes < 0).sum()) if episode_complete else None,
        draws=int((outcomes == 0).sum()) if episode_complete else None,
        score=float(outcomes.mean()) if episode_complete else None,
        perf=float((outcomes.mean() + 1) / 2) if episode_complete else None,
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

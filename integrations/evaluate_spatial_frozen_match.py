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
from integrations.spatial_action_sampling import (acting_logits, public_neutral_route_bonus,
                                                  public_owned_split_bias, public_safe_owned_split_bias,
                                                  public_weak_owned_route_penalty,
                                                  public_doomed_attack_route_penalty)
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
    parser.add_argument("--acting-greedy", action="store_true",
                        help="Choose argmax after the route/split transform used by PPO")
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--split-sampling-temperature", type=float,
                        help="Sample route at --sampling-temperature and full/half conditionally at this temperature")
    parser.add_argument("--half-logit-bias", type=float, default=0.0,
                        help="Diagnostic: add this offset to learner half-move logits before greedy selection")
    parser.add_argument("--expansion-audit", action="store_true",
                        help="Count first-episode move destinations and territory/army margins at fixed turns")
    parser.add_argument("--neutral-route-bias", type=float, default=0.0,
                        help="Diagnostic sampled-logit bonus for moves into visible empty neutral cells")
    parser.add_argument("--owned-split-bias", type=float, default=0.0,
                        help="Diagnostic conditional half-move bonus on owned routes from stacks >=5")
    parser.add_argument("--safe-owned-split-bias", type=float, default=0.0,
                        help="Diagnostic half-move bonus on interior owned routes from stacks of 5–19")
    parser.add_argument("--weak-owned-route-penalty", type=float, default=0.0,
                        help="Diagnostic route penalty for small-stack owned moves before 15 owned tiles")
    parser.add_argument("--doomed-attack-route-penalty", type=float, default=0.0,
                        help="Diagnostic route penalty when full army cannot capture a visible enemy")
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2 or args.pool_size <= 0:
        raise ValueError("Require a positive even game count and positive pool size")
    if not np.isfinite(args.sampling_temperature) or args.sampling_temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    if args.acting_greedy and args.sample_seed is not None:
        raise ValueError("Acting-greedy and sampled actions are separate modes")
    if args.acting_greedy and args.split_sampling_temperature is None:
        raise ValueError("Acting-greedy requires the structured route/split transform")
    if args.sampling_temperature != 1 and args.sample_seed is None and not args.acting_greedy:
        raise ValueError("Nondefault temperature requires sampled or acting-greedy actions")
    if args.split_sampling_temperature is not None and (
            (args.sample_seed is None and not args.acting_greedy) or not np.isfinite(args.split_sampling_temperature)
            or args.split_sampling_temperature <= 0):
        raise ValueError("Split temperature requires sampled or acting-greedy actions and a positive finite value")
    if not np.isfinite(args.half_logit_bias) or (args.half_logit_bias and (args.sample_seed is not None or args.acting_greedy)):
        raise ValueError("Half-logit bias requires greedy learner actions")
    if not np.isfinite(args.neutral_route_bias) or args.neutral_route_bias < 0 or (
            args.neutral_route_bias and args.sample_seed is None and not args.acting_greedy):
        raise ValueError("Neutral route bias requires sampled or acting-greedy actions and a finite nonnegative value")
    if not np.isfinite(args.owned_split_bias) or args.owned_split_bias < 0 or (
            args.owned_split_bias and args.split_sampling_temperature is None):
        raise ValueError("Owned split bias requires structured actions and a finite nonnegative value")
    if not np.isfinite(args.safe_owned_split_bias) or args.safe_owned_split_bias < 0 or (
            args.safe_owned_split_bias and args.split_sampling_temperature is None):
        raise ValueError("Safe owned split bias requires structured actions and a finite nonnegative value")
    if args.owned_split_bias and args.safe_owned_split_bias:
        raise ValueError("Use only one owned split diagnostic at a time")
    if not np.isfinite(args.weak_owned_route_penalty) or args.weak_owned_route_penalty < 0 or (
            args.weak_owned_route_penalty and args.split_sampling_temperature is None):
        raise ValueError("Weak owned route penalty requires structured actions and a finite nonnegative value")
    if not np.isfinite(args.doomed_attack_route_penalty) or args.doomed_attack_route_penalty < 0 or (
            args.doomed_attack_route_penalty and args.split_sampling_temperature is None):
        raise ValueError("Doomed attack route penalty requires structured actions and a finite nonnegative value")
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
    for k in ("frozen_bundle", "frozen_bundles", "frozen_build", "frozen_training", "frozen_checkpoint", "frozen_sha256"):
        options.pop(k, None)
    options.update(parallel_games=args.games, coworld_pool_size=args.pool_size,
                   shaping_weight=0.0, reward_scale=1.0, land_gain_reward_weight=0.0,
                   terminal_reward_mode="signed")
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
    action_counts = np.zeros(3, np.int64)
    action_disagreements = np.zeros(2, np.int64)
    destination_counts = np.zeros(4, np.int64)
    checkpoints = (25, 50, 100, 150, 200)
    progress = {}

    @jax.jit
    def audit_destinations(states, sides, indices):
        cells = 21 * 21
        passing = indices == 8 * cells
        route = indices % (4 * cells)
        source = route % cells
        direction = route // cells
        row, col = source // 21, source % 21
        dr = jnp.take(jnp.array((-1, 1, 0, 0)), direction)
        dc = jnp.take(jnp.array((0, 0, -1, 1)), direction)
        dest_row = jnp.clip(row + dr, 0, 20)
        dest_col = jnp.clip(col + dc, 0, 20)
        rows = jnp.arange(indices.shape[0])
        own = states.ownership[rows, sides, dest_row, dest_col]
        neutral = states.ownership_neutral[rows, dest_row, dest_col]
        return jnp.where(passing, 3, jnp.where(own, 0, jnp.where(neutral, 1, 2))).astype(jnp.int8)

    @jax.jit
    def audit_progress(states, sides):
        rows = jnp.arange(sides.shape[0])
        land = jnp.sum(states.ownership, axis=(2, 3), dtype=jnp.int32)
        army = jnp.sum(states.armies[:, None] * states.ownership, axis=(2, 3), dtype=jnp.int32)
        return (land[rows, sides] - land[rows, 1 - sides],
                army[rows, sides] - army[rows, 1 - sides])
    try:
        values, masks = env.reset_device(f"{args.seed}:0:0")
        sides = np.asarray(env.sides)
        device_sides = jnp.asarray(sides)
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
            outputs = np.asarray(forward(values)).copy()
            assert outputs.shape == (args.games, 3530) and np.isfinite(outputs).all()
            legal = np.asarray(masks, bool)
            raw_greedy = np.argmax(np.where(legal, outputs[:, :3529], -np.inf), axis=1)
            if args.half_logit_bias:
                outputs[:, 1764:3528] += args.half_logit_bias
            biased_greedy = np.argmax(np.where(legal, outputs[:, :3529], -np.inf), axis=1)
            if args.sample_seed is None:
                if args.acting_greedy:
                    split_bias = (public_safe_owned_split_bias(values, args.safe_owned_split_bias, np)
                                  if args.safe_owned_split_bias else
                                  public_owned_split_bias(values, args.owned_split_bias, np)
                                  if args.owned_split_bias else None)
                    logits = np.asarray(acting_logits(outputs, args.sampling_temperature,
                                                      args.split_sampling_temperature, np,
                                                      split_bias)[:, :3529])
                    if args.neutral_route_bias:
                        logits += public_neutral_route_bonus(values, args.neutral_route_bias, np)
                    if args.weak_owned_route_penalty:
                        logits += public_weak_owned_route_penalty(np.asarray(values), args.weak_owned_route_penalty, np)
                    if args.doomed_attack_route_penalty:
                        logits += public_doomed_attack_route_penalty(np.asarray(values), args.doomed_attack_route_penalty, np)
                    chosen = np.argmax(np.where(legal, logits, -np.inf), axis=1)
                else:
                    chosen = biased_greedy
            else:
                key = jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn)
                split_bias = (jnp.asarray(public_safe_owned_split_bias(
                                  np.asarray(values), args.safe_owned_split_bias, np))
                              if args.safe_owned_split_bias else
                              jnp.asarray(public_owned_split_bias(np.asarray(values), args.owned_split_bias, np))
                              if args.owned_split_bias else None)
                logits = (acting_logits(jnp.asarray(outputs), args.sampling_temperature,
                                       args.split_sampling_temperature, jnp, split_bias)[:, :3529]
                          if args.split_sampling_temperature is not None
                          else jnp.asarray(outputs[:, :3529]) / args.sampling_temperature)
                if args.neutral_route_bias:
                    logits += jnp.asarray(public_neutral_route_bonus(
                        np.asarray(values), args.neutral_route_bias, np))
                if args.weak_owned_route_penalty:
                    logits += jnp.asarray(public_weak_owned_route_penalty(
                        np.asarray(values), args.weak_owned_route_penalty, np))
                if args.doomed_attack_route_penalty:
                    logits += jnp.asarray(public_doomed_attack_route_penalty(
                        np.asarray(values), args.doomed_attack_route_penalty, np))
                chosen = np.asarray(sample_flat_logits(
                    key, logits, jnp.asarray(legal),
                ))
            actions = chosen.astype(np.int32)[:, None]
            assert legal[np.arange(args.games), actions[:, 0]].all()
            action_counts += np.bincount(np.where(chosen < 1764, 0,
                                                  np.where(chosen < 3528, 1, 2))[~finished], minlength=3)
            chosen_route = np.where(chosen == 3528, 1764, chosen % 1764)
            greedy_route = np.where(raw_greedy == 3528, 1764, raw_greedy % 1764)
            action_disagreements[0] += np.count_nonzero((chosen_route != greedy_route) & ~finished)
            action_disagreements[1] += np.count_nonzero((chosen_route == greedy_route)
                                                        & (chosen != raw_greedy) & ~finished)
            if args.expansion_audit:
                destinations = np.asarray(audit_destinations(env.states, device_sides, jnp.asarray(chosen)))
                destination_counts += np.bincount(destinations[~finished], minlength=4)
                if turn in checkpoints:
                    land_margin, army_margin = map(np.asarray, audit_progress(env.states, device_sides))
                    progress[str(turn)] = dict(games=int((~finished).sum()),
                                               land_margin_sum=int(land_margin[~finished].sum()),
                                               army_margin_sum=int(army_margin[~finished].sum()))
            if turn == 0:
                reference = policy.forward(np.asarray(values))
                if args.half_logit_bias:
                    reference[:, 1764:3528] += args.half_logit_bias
                assert np.allclose(outputs, reference, rtol=2e-5, atol=2e-5)
                if args.sample_seed is None and not args.acting_greedy:
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
                  action_selection="argmax_acting" if args.acting_greedy else "argmax" if args.sample_seed is None else "sample",
                  sample_seed=args.sample_seed,
                  sampling_temperature=args.sampling_temperature if (args.sample_seed is not None or args.acting_greedy) else None,
                  split_sampling_temperature=args.split_sampling_temperature,
                  half_logit_bias=args.half_logit_bias,
                  neutral_route_bias=args.neutral_route_bias,
                  owned_split_bias=args.owned_split_bias,
                  safe_owned_split_bias=args.safe_owned_split_bias,
                  weak_owned_route_penalty=args.weak_owned_route_penalty,
                  doomed_attack_route_penalty=args.doomed_attack_route_penalty,
                  first_episode_actions=dict(full=int(action_counts[0]), half=int(action_counts[1]),
                                             pass_actions=int(action_counts[2])),
                  first_episode_vs_raw_greedy=dict(route_changes=int(action_disagreements[0]),
                                                   split_changes=int(action_disagreements[1])),
                  opponent_action_selection="argmax",
                  checkpoint_sha256=hashlib.sha256((args.bundle / "policy.bin").read_bytes()).hexdigest(),
                  opponent_sha256=hashlib.sha256((args.opponent_bundle / "policy.bin").read_bytes()).hexdigest(),
                  episode_limit=env.horizon,
                  coworld_classic_rules=env.base.env.coworld_classic_rules,
                  pool_generation=int(env._pool_generation),
                  wins=int((outcomes > 0).sum()), losses=int((outcomes < 0).sum()), draws=int((outcomes == 0).sum()),
                  score=float(outcomes.mean()), turns=turn + 1, wall_seconds=time.monotonic() - start)
    if args.expansion_audit:
        result["expansion_audit"] = dict(
            destination_counts=dict(own=int(destination_counts[0]), neutral=int(destination_counts[1]),
                                    enemy=int(destination_counts[2]), passes=int(destination_counts[3])),
            checkpoints=progress,
            scope="Post-game omniscient audit of first-episode actions and state; no hidden information fed to actors",
        )
    np.save(args.output / "outcomes.npy", outcomes)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

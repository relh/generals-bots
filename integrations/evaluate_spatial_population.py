"""Held-out first-episode Classic results by frozen/scripted opponent and seat.

This evaluates immutable portable actors with their exported serving sampler.
It does not train, modify the opponent pool, or use hidden game state to act.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.spatial_destination_audit import destination_categories
from integrations.spatial_policy_bundle import SpatialPlayerPolicy
from integrations.spatial_selfplay import SpatialPopulationOpponentPufferEnvironment, frozen_action_indices


def summarize(labels, sides, outcomes, names):
    """Keep seat effects visible instead of merging unequal opponent groups."""
    result = {}
    for index, name in enumerate(names):
        result[name] = {}
        for side in (0, 1):
            values = outcomes[(labels == index) & (sides == side)]
            if not len(values):
                raise ValueError(f"Missing opponent {name} on side {side}")
            result[name][str(side)] = dict(
                games=int(len(values)), wins=int((values == 1).sum()),
                losses=int((values == -1).sum()), draws=int((values == 0).sum()),
                score=float(values.mean()),
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--population-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, required=True)
    parser.add_argument("--destination-audit", action="store_true",
                        help="Record first-episode destination types by game phase")
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2 or args.pool_size <= 0:
        raise ValueError("Require a positive even game count and positive pool size")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Population strength evaluation requires GPU execution")
    policy = SpatialPlayerPolicy(args.bundle)
    if policy.action_mode != "structured_sample":
        raise ValueError("Population evaluation requires the exact structured serving sampler")
    build = json.loads(args.population_build.read_text())
    environment = build["config"]["python_environment"]
    if environment["factory"] != "integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment":
        raise ValueError("Expected a pinned Classic spatial population build")
    opponent_fields = {"frozen_bundle", "frozen_bundles", "opponent_weights",
                       "scripted_opponents", "classic_siege_workers"}
    options = {key: value for key, value in environment["options"].items() if key in opponent_fields}
    actors = [policy] + [SpatialPlayerPolicy(Path(bundle)) for bundle in options["frozen_bundles"]]
    if any(args.seed in actor.asset.metadata["training_seeds"] for actor in actors):
        raise ValueError("Evaluation seed must be absent from all training lineages")
    options.update(parallel_games=args.games, coworld_pool_size=args.pool_size, balance_opponent_sides=True,
                   shaping_weight=0.0, reward_scale=1.0, terminal_reward_mode="signed",
                   coworld_position_probability=0.0)
    args.output.mkdir(parents=True, exist_ok=False)
    context = EnvironmentContext(seed=args.seed, index=0, mode="train", output=args.output)
    env = SpatialPopulationOpponentPufferEnvironment(context=context, **options)

    @jax.jit
    def choose(values, masks, keys):
        with jax.default_matmul_precision("highest"):
            outputs = policy._forward(values[:, :policy.observation_size], jnp)
        return frozen_action_indices(policy, outputs, masks, keys, values)

    start = time.monotonic()
    finished = np.zeros(args.games, bool)
    outcomes = np.zeros(args.games, np.float32)
    destination_counts = np.zeros((args.games, 3, 4), np.int32) if args.destination_audit else None
    try:
        values, masks = env.reset_device(f"{args.seed}:0:0")
        sides = np.asarray(env.sides)
        labels = env._population_labels.copy()
        names = tuple("frozen_" + digest[:12] for digest in env._population_checksums) + env._population_script_names
        if (sides == 0).sum() != args.games // 2 or (sides == 1).sum() != args.games // 2:
            raise ValueError("Learner seats are not balanced")
        for name in range(len(names)):
            if not ((labels == name) & (sides == 0)).any() or not ((labels == name) & (sides == 1)).any():
                raise ValueError("Population opponent absent from a seat")
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(env.states)]
        hashes = []
        for row in range(args.games):
            digest = hashlib.sha256()
            for leaf in leaves:
                digest.update(str((leaf.dtype.str, leaf.shape[1:])).encode())
                digest.update(leaf[row].tobytes())
            hashes.append(digest.hexdigest())
        np.save(args.output / "initial_state_sha256.npy", np.asarray(hashes, dtype="U64"))
        np.save(args.output / "initial_sides.npy", sides)
        np.save(args.output / "opponent_labels.npy", labels)
        for turn in range(env.horizon):
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.sample_seed), turn), args.games)
            chosen = np.asarray(choose(values, masks, keys))
            legal = np.asarray(masks, bool)
            if not legal[np.arange(args.games), chosen].all():
                raise ValueError("Candidate sampled an illegal action")
            if destination_counts is not None:
                active_rows = np.flatnonzero(~finished)
                categories = np.asarray(destination_categories(
                    env.states.ownership_neutral, env.states.ownership,
                    env.sides, jnp.asarray(chosen), jnp.asarray(~finished),
                ))
                selected = categories[active_rows]
                if not np.isin(selected, (1, 2, 3, 4)).all():
                    raise ValueError("Active legal action lacks a destination category")
                np.add.at(destination_counts, (active_rows, min(turn // 100, 2), selected - 1), 1)
            values, masks, rewards, done, _ = env.step_device(jnp.asarray(chosen[:, None]))
            ended = np.asarray(done, bool) & ~finished
            reward = np.asarray(rewards)
            if not np.isfinite(reward).all() or not np.isin(reward[ended], (-1, 0, 1)).all():
                raise ValueError("Population outcome is nonfinite or not signed terminal score")
            outcomes[ended] = reward[ended]
            finished |= ended
            if turn % 100 == 0:
                print(json.dumps(dict(turn=turn + 1, finished=int(finished.sum()))), flush=True)
            if finished.all():
                break
        if not finished.all():
            raise ValueError("Some first episodes did not terminate within the Classic cap")
        result = dict(scope="Held-out first episodes by public-view opponent and learner seat",
                      checkpoint_sha256=policy.asset.metadata["policy_sha256"],
                      training_seeds=policy.asset.metadata["training_seeds"],
                      opponent_training_seeds=[actor.asset.metadata["training_seeds"] for actor in actors[1:]],
                      population_build_sha256=hashlib.sha256(args.population_build.read_bytes()).hexdigest(),
                      frozen_policy_sha256=env._population_checksums,
                      opponent_weights=env._population_weights,
                      coworld_classic_rules=env.base.env.coworld_classic_rules,
                      games=args.games, pool_size=args.pool_size, seed=args.seed,
                      sample_seed=args.sample_seed, unique_initial_states=len(set(hashes)),
                      action_selection=dict(mode=policy.action_mode,
                                            move_temperature=policy.move_temperature,
                                            early_route_temperature=policy.early_route_temperature,
                                            early_route_turns=policy.early_route_turns,
                                            split_temperature=policy.split_temperature,
                                            route_half_weight=policy.route_half_weight,
                                            full_action_temperature=policy.full_action_temperature,
                                            neutral_route_bias=policy.neutral_route_bias,
                                            weak_owned_route_penalty=policy.weak_owned_route_penalty,
                                            doomed_attack_route_penalty=policy.doomed_attack_route_penalty),
                      by_opponent_and_seat=summarize(labels, sides, outcomes, names),
                      wins=int((outcomes == 1).sum()), losses=int((outcomes == -1).sum()),
                      draws=int((outcomes == 0).sum()), turns=turn + 1,
                      wall_seconds=time.monotonic() - start)
        np.save(args.output / "outcomes.npy", outcomes)
        if destination_counts is not None:
            np.save(args.output / "destination_counts.npy", destination_counts)
            result["destination_audit"] = dict(
                scope="Omniscient post-action audit; destination ownership was never fed to the policy",
                phases=("turns_0_99", "turns_100_199", "turns_200_plus"),
                categories=("neutral", "owned", "enemy", "pass"),
                counts=destination_counts.sum(axis=0).tolist(),
                by_opponent_and_seat={
                    name: {str(side): destination_counts[(labels == index) & (sides == side)].sum(axis=0).tolist()
                           for side in (0, 1)}
                    for index, name in enumerate(names)
                },
            )
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

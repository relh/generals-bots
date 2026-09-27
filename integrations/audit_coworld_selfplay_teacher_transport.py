"""Check two-seat GPU teacher targets against each seat's public action hint."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsSelfPlayPufferEnvironment
from integrations.puffer_codec import hinted_replay_indices


def check(values, masks, agents):
    cells = 21 * 21
    observation_size = 14 * cells
    move_size = 4 * cells + 1
    actions = move_size + 2
    data = np.asarray(values)
    legal = np.asarray(masks, dtype=bool)
    assert data.shape == (agents, observation_size + 2 * actions + 4)
    assert legal.shape == (agents, actions)
    public = data[:, :observation_size]
    probabilities = data[:, observation_size:observation_size + actions]
    carried_masks = data[:, observation_size + actions:observation_size + 2 * actions]
    weights = data[:, observation_size + 2 * actions:observation_size + 2 * actions + 2]
    np.testing.assert_array_equal(carried_masks, legal.astype(np.float32))
    assert np.all(data[:, -2:] == 0)
    hints = hinted_replay_indices(public, 21, channels=14)
    rows = np.arange(agents)
    moves = hints[:, 0] != move_size - 1
    assert legal[rows, hints[:, 0]].all()
    assert np.all(probabilities[rows, hints[:, 0]] == 1)
    assert np.all(probabilities[:, :move_size].sum(axis=1) == 1)
    assert np.all(weights[:, 0] == 1)
    np.testing.assert_array_equal(weights[:, 1], moves.astype(np.float32))
    np.testing.assert_array_equal(probabilities[:, move_size:].sum(axis=1), moves.astype(np.float32))
    if moves.any():
        move_rows = rows[moves]
        assert legal[move_rows, move_size + hints[moves, 1]].all()
        assert np.all(probabilities[move_rows, move_size + hints[moves, 1]] == 1)
    return hints, int(moves.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=32)
    args = parser.parse_args()
    assert args.turns > 0 and jax.devices()[0].platform == "gpu"
    manifest = json.loads(args.build.read_text())
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["coworld_classic"] and options["prior_hint_features"] and options["context_hint_features"]
    options.update(parallel_games=4, coworld_pool_size=16, opponent="expander_harvester",
                   deduplicate_opponent_branches=False, balance_opponent_sides=False,
                   supervise_teacher=True, teacher="expander_harvester")
    env = BatchedGeneralsSelfPlayPufferEnvironment(
        context=EnvironmentContext(seed=713, index=0, mode="train", output=Path("/tmp")), **options,
    )
    assert env.spec.agents == 8 and env.spec.teacher
    total_moves = 0
    distinct_seat_actions = 0
    try:
        values, masks = env.reset_device("selfplay-teacher-713")
        assert not np.array_equal(np.asarray(values[0, :6174]), np.asarray(values[1, :6174]))
        for _ in range(args.turns):
            hints, moves = check(values, masks, env.spec.agents)
            total_moves += moves
            distinct_seat_actions += int((hints[0::2, 0] != hints[1::2, 0]).sum())
            splits = np.where(hints[:, 1] >= 0, hints[:, 1], 0)
            proposed = jnp.asarray(np.stack((hints[:, 0], splits), axis=1), dtype=jnp.int32)
            values, masks, rewards, dones, episode_done = env.step_device(proposed)
            assert np.isfinite(np.asarray(rewards)).all() and episode_done is False
            done_pairs = np.asarray(dones).reshape(4, 2)
            np.testing.assert_array_equal(done_pairs[:, 0], done_pairs[:, 1])
        check(values, masks, env.spec.agents)
    finally:
        env.close()
    assert total_moves > 0 and distinct_seat_actions > 0
    print(json.dumps(dict(scope="Two-seat GPU teacher transport contract", turns=args.turns,
                          agents=8, labeled_moves=total_moves, distinct_seat_actions=distinct_seat_actions)), flush=True)


if __name__ == "__main__":
    main()

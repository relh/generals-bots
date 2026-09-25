"""Measure scripted teachers on the exact Coworld Classic map distribution."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from metta_training.environment import EnvironmentContext
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def evaluate(seed: int, games: int, teacher: str, opponent: str) -> dict:
    context = EnvironmentContext(seed=seed, index=0, mode="evaluate", output=Path("/tmp"))
    environment = BatchedGeneralsPufferEnvironment(
        context=context,
        parallel_games=games,
        opponent=opponent,
        teacher=teacher,
        sparse_teacher=True,
        factorized_actions=True,
        coworld_classic=True,
        coworld_pool_size=64,
        compact_features=True,
        lean_features=True,
    )
    environment.reset(f"seed-{seed}")
    sides, opponents, pool = environment.sides, environment.opponent_ids, environment.base.pool
    empty_cache = jnp.zeros((games, 5), dtype=jnp.int32)

    @jax.jit
    def episode(states, keys):
        finished = jnp.zeros((games,), dtype=bool)
        outcomes = jnp.zeros((games,), dtype=jnp.float32)

        def turn(_, carry):
            states, keys, finished, outcomes = carry
            actions = environment._teacher_actions(states, sides, keys)
            index = jnp.where(
                actions[:, 0] == 1,
                1764,
                actions[:, 3] * 441 + actions[:, 1] * 21 + actions[:, 2],
            )
            states, keys, _, _, _, done, reward, _ = environment._advance_states(
                states, pool, sides, opponents, index, actions[:, 4], keys, empty_cache, ~finished
            )
            return states, keys, finished | done, outcomes + reward

        return jax.lax.fori_loop(0, 1200, turn, (states, keys, finished, outcomes))[3]

    outcomes = np.asarray(episode(environment.states, environment.keys))
    environment.close()
    return {
        "seed": seed,
        "games": games,
        "teacher": teacher,
        "opponent": opponent,
        "wins": int(np.count_nonzero(outcomes > 0)),
        "losses": int(np.count_nonzero(outcomes < 0)),
        "draws": int(np.count_nonzero(outcomes == 0)),
        "performance": float((outcomes.mean() + 1) / 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[901, 902])
    parser.add_argument("--teacher", choices=["harvester", "expander_harvester", "sentinel"],
                        default="harvester")
    parser.add_argument("--opponent", choices=["mixed", "expander_harvester", "sentinel"],
                        default="mixed")
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("Scripted benchmark requires CUDA")
    print("GPU:", jax.devices("cuda")[0], flush=True)
    for seed in args.seeds:
        print(json.dumps(evaluate(seed, args.games, args.teacher, args.opponent)), flush=True)


if __name__ == "__main__":
    main()

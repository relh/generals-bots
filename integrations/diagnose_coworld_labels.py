"""Audit sparse Harvester labels and legal moves in the batched GPU environment."""

import argparse
from pathlib import Path

import jax
import numpy as np

from metta_training.environment import EnvironmentContext
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--turns", type=int, default=128)
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("Sparse label diagnosis requires CUDA")
    context = EnvironmentContext(seed=901, index=0, mode="train", output=Path("/tmp"))
    env = BatchedGeneralsPufferEnvironment(
        context=context, parallel_games=args.games, opponent="mixed", teacher="harvester",
        sparse_teacher=True, factorized_actions=True, coworld_classic=True,
        coworld_pool_size=64, compact_features=True, lean_features=True,
        packed_directional_features=True,
    )
    observation = env.reset("diagnose-901")
    for turn in range(args.turns):
        labels = np.asarray(observation.replay_metadata, dtype=np.int32)
        legal = np.asarray(observation.action_masks, dtype=bool)
        active = labels[:, 0] >= 0
        chosen = np.where(active, labels[:, 0], 1764)
        assert legal[np.arange(args.games), chosen].all()
        if turn in (0, 4, 8, 16, 32, 64, 127):
            print({
                "turn": turn, "labeled": int(active.sum()),
                "pass_labels": int(((labels[:, 0] == 1764) & active).sum()),
                "move_labels": int(((labels[:, 0] >= 0) & (labels[:, 0] < 1764)).sum()),
                "split_labels": int((labels[:, 1] >= 0).sum()),
            }, flush=True)
        actions = np.stack((chosen, np.where(labels[:, 1] >= 0, labels[:, 1], 0)), axis=1)
        observation = env.step(actions.tolist()).observation
    env.close()


if __name__ == "__main__":
    main()

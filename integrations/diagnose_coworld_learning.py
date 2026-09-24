"""Measure teacher-action likelihood on identical states across checkpoints."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from metta_training.environment import EnvironmentContext, NumericObservation
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--turns", type=int, default=16)
    args = parser.parse_args()
    if not jax.devices("cuda"):
        raise RuntimeError("Learning diagnosis requires CUDA")
    context = EnvironmentContext(seed=901, index=0, mode="train", output=Path("/tmp"))
    env = BatchedGeneralsPufferEnvironment(
        context=context, parallel_games=args.games, opponent="mixed", teacher="harvester",
        sparse_teacher=True, factorized_actions=True, coworld_classic=True,
        coworld_pool_size=64, compact_features=True, lean_features=True, hint_features=True,
    )
    observation = env.reset("diagnose-learning-901")
    for _ in range(args.turns):
        labels = np.asarray(observation.replay_metadata, dtype=np.int32)
        source = np.where(labels[:, 0] >= 0, labels[:, 0], 1764)
        split = np.where(labels[:, 1] >= 0, labels[:, 1], 0)
        observation = env.step(np.stack((source, split), axis=1).tolist()).observation
    labels = np.asarray(observation.replay_metadata, dtype=np.int32)
    values = np.asarray(observation.values, dtype=np.float32)
    masks = np.asarray(observation.action_masks, dtype=bool)
    assert np.all(labels[:, 0] >= 0)
    env.close()

    run = args.run / "checkpoints" / "metta_generals" / args.run.name
    for step in (524288, 1048576):
        checkpoint = run / f"{step:016d}.bin"
        policy = FrozenPolicy(FrozenPolicyConfig(
            device="cuda:0", build=args.build / "build.json", checkpoint=checkpoint,
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        ))
        policy.reset(f"diagnose-{step}")
        losses, matches, pass_predictions = [], 0, 0
        for index in range(args.games):
            prediction = policy.predict(0, NumericObservation(
                values=[values[index].tolist()], action_masks=[masks[index].tolist()]
            ))
            probabilities = np.asarray(prediction.probabilities)
            source = int(labels[index, 0])
            split = int(labels[index, 1])
            likelihood = probabilities[source]
            if split >= 0:
                likelihood *= probabilities[1765 + split]
            losses.append(-np.log(max(likelihood, 1e-30)))
            chosen = int(np.argmax(probabilities[:1765]))
            matches += chosen == source
            pass_predictions += chosen == 1764
        print(json.dumps({
            "step": step, "games": args.games, "mean_teacher_nll": float(np.mean(losses)),
            "teacher_source_top1": matches / args.games,
            "predicted_pass_fraction": pass_predictions / args.games,
            "teacher_pass_fraction": float(np.mean(labels[:, 0] == 1764)),
        }), flush=True)


if __name__ == "__main__":
    main()

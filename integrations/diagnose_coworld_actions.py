"""Trace frozen Classic policy action probabilities on its own rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from metta_training.environment import EnvironmentContext, NumericObservation
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--turns", type=int, default=120)
    args = parser.parse_args()
    manifest = json.loads(args.build.read_text())
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["factorized_actions"] and options["coworld_classic"]
    options["parallel_games"] = args.games
    context = EnvironmentContext(seed=1101, index=0, mode="evaluate", output=Path("/tmp"))
    env = BatchedGeneralsPufferEnvironment(context=context, **options)
    assert env.spec.observation_size == manifest["config"]["fabric"]["observation_size"]
    assert env.spec.action_sizes == manifest["config"]["fabric"]["action_sizes"]
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    policy = FrozenPolicy(FrozenPolicyConfig(
        device="cuda:0", build=args.build, checkpoint=args.checkpoint, sha256=digest,
    ))
    policy.reset("classic-action-diagnosis-1101")
    observation = env.reset("classic-action-diagnosis-1101")
    records = []
    pass_index = env.spec.action_sizes[0] - 1
    try:
        for turn in range(args.turns):
            values = np.asarray(observation.values)
            masks = np.asarray(observation.action_masks, dtype=bool)
            actions = []
            for seat in range(args.games):
                prediction = policy.predict(seat, NumericObservation(
                    values=[values[seat].tolist()], action_masks=[masks[seat].tolist()],
                ))
                probabilities = np.asarray(prediction.probabilities)
                head = probabilities[:pass_index + 1]
                chosen = int(np.argmax(head))
                split = int(np.argmax(probabilities[pass_index + 1:]))
                actions.append([chosen, split])
                if not env.finished[seat]:
                    records.append((turn, seat, int(masks[seat, :pass_index].sum()),
                                    float(head[pass_index]), chosen == pass_index,
                                    float(head.max()), split,
                                    float(probabilities[pass_index + 2])))
            observation = env.step(actions).observation
            if (turn + 1) % 20 == 0:
                recent = [row for row in records if row[0] >= turn - 19]
                movable = [row for row in recent if row[2] > 0]
                print(json.dumps({
                    "turn": turn + 1, "states": len(recent),
                    "legal_moves_mean": float(np.mean([r[2] for r in recent])),
                    "pass_probability_mean": float(np.mean([r[3] for r in recent])),
                    "pass_argmax_fraction": float(np.mean([r[4] for r in recent])),
                    "pass_argmax_with_moves_fraction": float(np.mean([r[4] for r in movable])) if movable else None,
                    "top_probability_mean": float(np.mean([r[5] for r in recent])),
                    "split_argmax_fraction": float(np.mean([r[6] for r in recent])),
                    "split_probability_mean": float(np.mean([r[7] for r in recent])),
                    "completed_games": int(np.sum(env.completed)),
                }), flush=True)
    finally:
        env.close()
    print(json.dumps({"checkpoint_sha256": digest, "states": len(records),
                      "completed_games": int(np.sum(env.completed))}), flush=True)


if __name__ == "__main__":
    main()

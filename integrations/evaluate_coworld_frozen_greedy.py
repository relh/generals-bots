"""Evaluate the hosted argmax policy on verified, held-out Classic games."""

import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--opponent", choices=("expander_harvester", "sentinel", "strong_mixed"), required=True)
    args = parser.parse_args()
    assert args.games > 0 and jax.devices()[0].platform == "gpu"
    training = TrainingRecord.model_validate_json((args.run / "training.json").read_text())
    assert args.seed not in training_lineage_seeds(args.run, training)
    manifest = json.loads(args.build.read_text())
    assert manifest["model_sha256"] == training.build.model_sha256
    policy = FrozenPolicy(FrozenPolicyConfig(
        build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0",
    ))
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["coworld_classic"] and not options["teacher_rollouts"]
    options.update(parallel_games=args.games, opponent=args.opponent, supervise_teacher=False)
    args.output.mkdir(parents=True, exist_ok=False)
    context = EnvironmentContext(seed=args.seed, index=0, mode="evaluate", output=args.output)
    env = BatchedGeneralsPufferEnvironment(context=context, **options)
    reset_seed = f"{args.seed}:0:0"
    policy.reset(reset_seed)
    seats = list(range(args.games))
    start = time.monotonic()
    try:
        observation = env.reset(reset_seed)
        for turn in range(env.horizon):
            predictions = policy.predict_many(seats, observation)
            actions = np.asarray([
                (np.argmax(prediction.probabilities[:1765]), np.argmax(prediction.probabilities[1765:]))
                for prediction in predictions
            ], dtype=np.int32)
            masks = np.asarray(observation.action_masks, dtype=bool)
            assert masks[np.arange(args.games), actions[:, 0]].all()
            assert masks[np.arange(args.games), 1765 + actions[:, 1]].all()
            transition = env.step(actions)
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
        scope="Frozen GPU argmax inference on Classic maps; hosted service startup remains separate",
        action_selection="argmax_per_head", seed=args.seed, reset_seed=reset_seed,
        games=args.games, opponent=args.opponent, checkpoint_sha256=args.sha256,
        model_sha256=manifest["model_sha256"], options=options,
        wins=int((outcomes > 0).sum()), losses=int((outcomes < 0).sum()), draws=int((outcomes == 0).sum()),
        score=float(outcomes.mean()), perf=float((outcomes.mean() + 1) / 2),
        turns=turn + 1, wall_seconds=time.monotonic() - start,
    )
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(args.output / "outcomes.npy", outcomes)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

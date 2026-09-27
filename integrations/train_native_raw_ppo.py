"""Bounded reward-only Puffer5 training on public Coworld Classic observations."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from metta_training.puffer import RunConfig, train_puffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=8_388_608)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    args = parser.parse_args()
    if (args.timesteps <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or args.checkpoint_interval <= 0):
        raise ValueError("Invalid training budget or learning rate")
    build = json.loads((args.build / "build.json").read_text())
    options = build["config"]["python_environment"]["options"]
    if (build["config"]["native_hint_prior"] is not None or options["hint_features"]
            or options["compact_features"] or options["prior_hint_features"]
            or options["supervise_teacher"] or options["teacher_rollouts"]
            or options["shaping_gamma"] != .999 or not options["balance_opponent_sides"]
            or options["opponent"] != "strong_mixed"):
        raise ValueError("Unexpected raw RL environment")
    record = json.loads(args.template.read_text())
    record["total_timesteps"] = args.timesteps
    record["seed"] = args.seed
    record["initialize"] = None
    record["native_policy_initializer"] = None
    record["overrides"].update({
        "vec.total_agents": 65536, "vec.num_buffers": 1, "vec.num_threads": 1,
        "base.cudagraphs": -1, "base.checkpoint_interval": args.checkpoint_interval,
        "policy.hidden_size": 512, "policy.num_layers": 0,
        "train.learning_rate": args.learning_rate, "train.ent_coef": .01,
        "train.replay_ratio": 1.0, "train.horizon": 16,
        "train.minibatch_size": 524288,
    })
    assert record["overrides"]["train.gamma"] == .999
    config = RunConfig.model_validate(record)
    (args.output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    result = train_puffer(args.build, args.output / "run", config)
    assert result.trained_timesteps == args.timesteps
    checkpoint = args.output / "run" / result.final_checkpoint
    data = checkpoint.read_bytes()
    assert len(data) == (512 * 6174 + 1768 * 512) * 4
    assert all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data))
    print("NATIVE_RAW_PPO_PILOT_OK", result.trained_timesteps,
          hashlib.sha256(data).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

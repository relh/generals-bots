"""Bounded Puffer5 training from a fixed public-hint prior and weak trainable seed."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from metta_training.puffer import NativePolicyInitialization, RunConfig, train_puffer

from integrations.native_hint_initializer import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=8_388_608)
    parser.add_argument("--seed", type=int, default=1394)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--normalize-advantages", action="store_true")
    parser.add_argument("--total-agents", type=int, default=65536)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    args = parser.parse_args()
    if (
        args.timesteps <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate < 0
        or not math.isfinite(args.replay_ratio)
        or args.replay_ratio <= 0
        or args.checkpoint_interval <= 0
        or args.total_agents <= 0
        or args.horizon <= 0
        or not math.isfinite(args.gae_lambda)
        or not 0 < args.gae_lambda <= 1
    ):
        raise ValueError("Invalid training budget or learning rate")
    build = json.loads((args.build / "build.json").read_text())
    if args.normalize_advantages and "norm_adv = 0" not in (args.build / "source/config/default.ini").read_text():
        raise ValueError("Native advantage normalization requires its verified build")
    options = build["config"]["python_environment"]["options"]
    if (
        build["config"]["native_hint_prior"] is None
        or options["shaping_gamma"] != 0.999
        or not options["balance_opponent_sides"]
        or options["opponent"] != "strong_mixed"
    ):
        raise ValueError("Unexpected fixed-prior training environment")
    assert not (args.output / "run").exists()
    _, initializer = export(args.build / "build.json", args.output / "initializer", scale=0.001, layers=0)
    artifact = args.output / "initializer/initializer.json"
    record = json.loads(args.template.read_text())
    record["total_timesteps"] = args.timesteps
    record["seed"] = args.seed
    record["initialize"] = None
    record["native_policy_initializer"] = dict(
        manifest=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    record["overrides"].update(
        {
            "vec.total_agents": args.total_agents,
            "vec.num_buffers": 1,
            "vec.num_threads": 1,
            "base.cudagraphs": -1,
            "base.checkpoint_interval": args.checkpoint_interval,
            "policy.hidden_size": 512,
            "policy.num_layers": 0,
            "train.learning_rate": args.learning_rate,
            "train.anneal_lr": 0,
            "train.ent_coef": 0.0,
            "train.replay_ratio": args.replay_ratio,
            "train.horizon": args.horizon,
            "train.gae_lambda": args.gae_lambda,
            "train.minibatch_size": 524288,
        }
    )
    if args.normalize_advantages:
        record["overrides"]["train.norm_adv"] = 1
    assert record["overrides"]["train.gamma"] == 0.999
    config = RunConfig.model_validate(record)
    assert isinstance(config.native_policy_initializer, NativePolicyInitialization)
    (args.output / "config.json").write_text(config.model_dump_json(indent=2) + "\n")
    result = train_puffer(args.build, args.output / "run", config)
    assert result.trained_timesteps == args.timesteps
    copied_sha = hashlib.sha256((args.output / "run/initial-policy.bin").read_bytes()).hexdigest()
    assert copied_sha == initializer["policy_sha256"]
    checkpoint = args.output / "run" / result.final_checkpoint
    data = checkpoint.read_bytes()
    assert len(data) == initializer["parameter_count"] * 4
    assert all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data))
    print("NATIVE_FIXED_PRIOR_PILOT_OK", result.trained_timesteps, hashlib.sha256(data).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

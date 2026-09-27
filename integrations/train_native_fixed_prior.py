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
    args = parser.parse_args()
    if args.timesteps <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate < 0:
        raise ValueError("Invalid training budget or learning rate")
    build = json.loads((args.build / "build.json").read_text())
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
            "vec.total_agents": 65536,
            "vec.num_buffers": 1,
            "vec.num_threads": 1,
            "base.cudagraphs": -1,
            "base.checkpoint_interval": 1,
            "policy.hidden_size": 512,
            "policy.num_layers": 0,
            "train.learning_rate": args.learning_rate,
            "train.anneal_lr": 0,
            "train.ent_coef": 0.0,
            "train.replay_ratio": 1.0,
            "train.horizon": 16,
            "train.minibatch_size": 524288,
        }
    )
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

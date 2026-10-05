"""Bounded native Puffer5 probe from an explicit untrained public-hint artifact."""

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
    parser.add_argument("--logit-scale", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=1346)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--timesteps", type=int, default=33_554_432)
    parser.add_argument("--layers", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.timesteps <= 0:
        raise ValueError("Timesteps must be positive")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("Entropy coefficient must be finite and nonnegative")
    assert not (args.output / "run").exists()
    _, initializer = export(
        args.build / "build.json", args.output / "initializer", scale=args.logit_scale, layers=args.layers
    )
    artifact = args.output / "initializer/initializer.json"
    record = json.loads(args.template.read_text())
    record["total_timesteps"] = args.timesteps
    record["seed"] = args.seed
    record["initialize"] = None
    record["native_policy_initializer"] = dict(
        manifest=str(artifact), sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    record["overrides"].update({
        "vec.total_agents": 65536, "vec.num_buffers": 1, "vec.num_threads": 1,
        "base.cudagraphs": -1, "base.checkpoint_interval": 16,
        "policy.hidden_size": 512, "policy.num_layers": args.layers,
        "train.learning_rate": .003, "train.ent_coef": args.entropy_coef, "train.replay_ratio": 1.,
        "train.horizon": 16, "train.minibatch_size": 524288,
    })
    assert record["overrides"]["train.gamma"] == .999
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
    print("NATIVE_PARAMETERS_FINITE", len(data) // 4, hashlib.sha256(data).hexdigest(), flush=True)


if __name__ == "__main__":
    main()

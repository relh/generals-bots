"""Evaluate a deliberate Fabric model change with a verified old checkpoint."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from metta_training.native_build import environment_runtime, fabric_runtime
from metta_training.puffer import BuildManifest, TrainingRecord


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    manifest = BuildManifest.model_validate_json((args.build / "build.json").read_text())
    training = TrainingRecord.model_validate_json((args.run / "run.json").read_text())
    assert manifest.config.fabric is not None
    assert manifest.config.fabric.options["normalized_global"] is True
    assert training.build.config.fabric is not None
    assert training.build.config.fabric.factory == manifest.config.fabric.factory
    assert training.build.config.fabric.options.get("normalized_global") is None
    assert training.build.model_sha256 != manifest.model_sha256
    assert training.build.model_state_words == manifest.model_state_words
    assert args.checkpoint.is_relative_to(args.run / "checkpoints")
    weights = args.checkpoint.read_bytes()
    assert hashlib.sha256(weights).hexdigest() == args.checkpoint_sha256
    assert hashlib.sha256((args.build / "puffer").read_bytes()).hexdigest() == manifest.binary_sha256
    assert args.seed != training.config.seed
    args.output.mkdir(parents=True, exist_ok=False)
    owned, environment = fabric_runtime(
        manifest.config.fabric, manifest.model_sha256, manifest.model_state_words
    )
    environment = environment | environment_runtime(
        manifest.config.python_environment, manifest.environment_sha256,
        args.output, args.seed, "evaluate",
    )
    overrides = {k: v for k, v in training.config.overrides.items() if not k.startswith("objective.")}
    overrides.update(owned)
    checkpoint = args.output / "checkpoint.bin"
    checkpoint.write_bytes(weights)
    command = [str(args.build / "puffer"), "eval", str(checkpoint), "--headless",
               *[f"--{key}={value}" for key, value in sorted(overrides.items()) if key != "base.eval_episodes"],
               f"--base.seed={args.seed}", "--base.eval_episodes=1"]
    with (args.output / "console.log").open("w") as log:
        subprocess.run(command, cwd=args.build / "source", env=environment,
                       stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    matches = re.findall(
        r"^CUDA_EVAL env=(\w+) score=(\S+) perf=(\S+) games=(\d+) params=(\d+)$",
        (args.output / "console.log").read_text(), re.MULTILINE,
    )
    assert len(matches) == 1 and matches[0][0] == manifest.config.environment
    _, score, perf, games, parameters = matches[0]
    # The native evaluator counts batched environment episodes. Each one
    # contains one game for every parallel agent in this environment.
    parallel_games = manifest.config.python_environment.spec.agents
    record = {
        "kind": "explicit_model_ablation",
        "source_model_sha256": training.build.model_sha256,
        "target_model_sha256": manifest.model_sha256,
        "checkpoint_sha256": args.checkpoint_sha256,
        "opponent": manifest.config.python_environment.options["opponent"],
        "seed": args.seed, "score": float(score), "perf": float(perf),
        "eval_batches": int(games), "parallel_games_per_batch": parallel_games,
        "effective_games": int(games) * parallel_games,
        "parameters": int(parameters),
    }
    (args.output / "ablation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

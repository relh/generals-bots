"""Shift only the initialized hint weights in a frozen Fabric checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.native_fabric import NativeFabricPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--target-strength", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.build.read_text())
    config = manifest["config"]["fabric"]
    source_strength = float(config["options"]["hint_prior_strength"])
    if source_strength <= 0 or not 0 <= args.target_strength < source_strength:
        raise ValueError("Ablation must reduce a positive hint prior")
    checkpoint = args.checkpoint.read_bytes()
    source_sha = hashlib.sha256(checkpoint).hexdigest()
    if source_sha != args.sha256:
        raise ValueError("Source checkpoint digest mismatch")
    altered = json.loads(json.dumps(config))
    altered["options"]["hint_prior_strength"] = args.target_strength
    with jax.default_device(jax.devices("cpu")[0]):
        source_model = NativeFabricPolicy(json.dumps(config))
        target_model = NativeFabricPolicy(json.dumps(altered))
        source_init = np.frombuffer(source_model.initialize(7), np.float32)
        target_init = np.frombuffer(target_model.initialize(7), np.float32)
    trained = np.frombuffer(checkpoint, np.float32)
    if source_init.shape != target_init.shape or trained.shape != source_init.shape:
        raise ValueError("Model parameter layout changed")
    delta = target_init - source_init
    changed = np.flatnonzero(delta != 0)
    if not 4 * 21 * 21 <= len(changed) <= 3 * 4 * 21 * 21:
        raise ValueError(f"Unexpected number of changed hint weights: {len(changed)}")
    if not np.isfinite(delta).all() or not np.isfinite(trained).all():
        raise ValueError("Nonfinite source or ablation weights")
    candidate = (trained + delta).astype(np.float32)
    if not np.isfinite(candidate).all():
        raise ValueError("Nonfinite ablation weights")
    if not np.array_equal(candidate[np.setdiff1d(np.arange(len(candidate)), changed)],
                          trained[np.setdiff1d(np.arange(len(candidate)), changed)]):
        raise ValueError("Ablation changed non-hint parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "checkpoint.bin"
    target.write_bytes(candidate.tobytes())
    record = {
        "kind": "diagnostic_hint_prior_ablation_not_training",
        "source_checkpoint_sha256": source_sha,
        "ablation_checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_strength": source_strength,
        "target_strength": args.target_strength,
        "changed_parameter_count": int(len(changed)),
        "parameter_count": int(trained.size),
        "delta_min": float(delta[changed].min()),
        "delta_max": float(delta[changed].max()),
    }
    (args.output / "ablation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

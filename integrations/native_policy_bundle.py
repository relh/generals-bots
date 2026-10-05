"""Portable native MinGRU checkpoints and a single-seat player adapter."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from integrations.native_puffer_policy import NativePufferPolicy


def export_bundle(build, training, checkpoint, sha256, output):
    # Validate dimensions, finite parameters and checkpoint identity before export.
    NativePufferPolicy(build, training, checkpoint, sha256)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, source in (("build.json", build), ("training.json", training), ("policy.bin", checkpoint)):
        data = source.read_bytes()
        (output / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    (output / "native-policy.json").write_text(json.dumps(dict(
        schema="puffer5-min-gru-v1", files=files,
    ), indent=2) + "\n")


class NativePlayerPolicy:
    def __init__(self, bundle):
        manifest = json.loads((bundle / "native-policy.json").read_text())
        if manifest["schema"] != "puffer5-min-gru-v1":
            raise ValueError("Unsupported native policy bundle")
        if set(manifest["files"]) != {"build.json", "training.json", "policy.bin"}:
            raise ValueError("Unexpected native bundle files")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Native bundle checksum mismatch: " + name)
        self.policy = NativePufferPolicy(bundle / "build.json", bundle / "training.json",
                                         bundle / "policy.bin", manifest["files"]["policy.bin"])
        self.reset("")

    def reset(self, seed):
        self.state = self.policy.initial_state(1)

    def predict(self, seat, observation):
        if seat != 0:
            raise ValueError("Native player supports one seat per process")
        values = np.asarray(observation.values, dtype=np.float32)
        mask = np.asarray(observation.action_masks, dtype=bool)
        if (values.shape != (1, self.policy.observation_size)
                or mask.shape != (1, self.policy.logit_size)):
            raise ValueError("Unexpected native player observation")
        start = 0
        for size in self.policy.action_sizes:
            if not mask[:, start:start + size].any():
                raise ValueError("An action head has no legal actions")
            start += size
        decoded, self.state = self.policy.forward(jax.numpy.asarray(values), self.state)
        output = np.asarray(decoded)[0]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite native player prediction")
        # Existing player selects argmax per head; normalize only legal logits.
        probabilities = []
        start = 0
        for size in self.policy.action_sizes:
            stop = start + size
            logits = np.where(mask[0, start:stop], output[start:stop], -np.inf)
            weights = np.exp(logits - np.max(logits))
            probabilities.extend((weights / weights.sum()).tolist())
            start = stop
        return SimpleNamespace(probabilities=probabilities)


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    export_bundle(args.build, args.training, args.checkpoint, args.sha256, args.output)


if __name__ == "__main__":
    main()

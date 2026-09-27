"""Deterministic native MinGRU initialization from existing public move hints.

This is an initialization artifact, not a completed learner run or champion.
The native actor represents source preference plus global direction preference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parameters(scale=24.0):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    hidden, cells = 512, 441
    encoder = np.zeros((hidden, 6174), np.float32)
    decoder = np.zeros((1768, hidden), np.float32)
    recurrent = (np.zeros((3 * hidden, hidden), np.float32),)
    # Source preference is shared across the four directions; direction
    # preference is shared across every source. Their sum selects both.
    for direction in range(4):
        for cell in range(cells):
            encoder[cell, (4 + direction) * cells + cell] = 1
            encoder[cells + direction, (4 + direction) * cells + cell] = 1
            decoder[direction * cells + cell, cell] = scale
            decoder[direction * cells + cell, cells + direction] = scale
    encoder[445, 3 * cells] = 1  # signed pass flag, calibrated .375
    encoder[446, 2 * cells] = 1  # signed split flag, calibrated .125
    # All zero recurrent matrices yield the same time-dependent offset in
    # every hidden unit. Equal row sums cancel it across move logits.
    decoder[1764, 445] = scale
    decoder[1764, 447] = scale
    # Opposite signed split logits; unused unit447 cancels their offset.
    decoder[1765, 446], decoder[1765, 447] = -scale, scale
    decoder[1766, 446], decoder[1766, 447] = scale, -scale
    return encoder, decoder, recurrent


def export(build, output, scale=24.0):
    manifest = json.loads(build.read_text())
    config = manifest["config"]
    env = config["python_environment"]
    spec, options = env["spec"], env["options"]
    if (manifest["revision"] != "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
            or config["fabric"] is not None or config["precision"] != "float32"
            or spec["observation_size"] != 6174 or spec["action_sizes"] != [1765, 2]
            or spec["teacher"] or spec["routing"] or spec["replay_metadata_size"]
            or not options["prior_hint_features"] or not options["expander_hint_features"]
            or not options["context_hint_features"] or options["move_hint_scale"] != .375
            or options["split_hint_scale"] != .125):
        raise ValueError("Unsupported public-hint native contract")
    output.mkdir(parents=True, exist_ok=False)
    weights = parameters(scale)
    policy = output / "policy.bin"
    with policy.open("wb") as handle:
        for tensor in (weights[0], weights[1], *weights[2]):
            handle.write(b"\0" * ((-handle.tell()) % 16))
            handle.write(tensor.astype("<f4").tobytes())
    result = dict(schema="generals-native-public-hint-initializer-v1",
                  provenance="Deterministic weights mapping existing public source/direction/pass/split hints",
                  build_sha256=hashlib.sha256(build.read_bytes()).hexdigest(),
                  policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                  hidden_size=512, num_layers=1, observation_size=6174, action_sizes=[1765, 2],
                  precision="float32", parameter_count=4852736, logit_scale=scale,
                  trained_steps=0, training_seeds=[], learner_state=False, release_eligible=False)
    assert policy.stat().st_size == result["parameter_count"] * 4
    (output / "initializer.json").write_text(json.dumps(result, indent=2) + "\n")
    return weights, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, result = export(args.build, args.output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

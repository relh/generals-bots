"""Bounded CUDA probe for large-board Fabric initialization and inference."""

import sys
import time
from pathlib import Path

import jax
import numpy as np

from metta_training.native_fabric import NativeFabricPolicy
from metta_training.puffer import BuildManifest


def main(build_path: Path, local_features: int, global_features: int) -> None:
    if not jax.devices("cuda"):
        raise RuntimeError("Fabric probe requires a CUDA device")
    manifest = BuildManifest.model_validate_json(build_path.read_text())
    assert manifest.config.fabric is not None
    options = manifest.config.fabric.options | {
        "features_per_site": local_features, "global_features": global_features,
        "input_radius": 0.1,
    }
    model = manifest.config.fabric.model_copy(update={"options": options})
    started = time.monotonic()
    policy = NativeFabricPolicy(model.model_dump_json())
    print(f"construct_seconds={time.monotonic() - started:.3f}", flush=True)
    parameters = policy.initialize(17)
    state = bytes(policy.state_words * 4)
    observation = np.zeros(policy.input_size, dtype=np.float32).tobytes()
    started = time.monotonic()
    outputs, _, _ = policy.forward(parameters, state, observation, bytes(4), 1, 1, True)
    assert np.isfinite(np.frombuffer(outputs, np.float32)).all()
    print(f"first_forward_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    policy.forward(parameters, state, observation, bytes(4), 1, 1, True)
    print(f"warm_forward_seconds={time.monotonic() - started:.3f}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

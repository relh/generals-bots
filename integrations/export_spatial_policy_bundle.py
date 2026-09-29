"""Export immutable spatial weights using the verified realized graph layout."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def export_bundle(build, training, checkpoint, sha256, factory_source, output):
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from integrations.direct_spatial_optimization import DirectSpatial
    from integrations.memoryless_optimization import BRIDGE_SHA256, verify_configuration
    import metta_training.native_fabric as native_module
    from metta_training.native_fabric import NativeFabricPolicy

    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Native bridge differs from the verified parameter layout")

    manifest = json.loads(build.read_text())
    record = json.loads(training.read_text())
    if record["build"] != manifest:
        raise ValueError("Spatial training/build manifests differ")
    configuration = json.dumps(manifest["config"]["fabric"])
    verify_configuration(configuration)
    policy = NativeFabricPolicy(configuration)
    model = DirectSpatial(policy)
    raw = checkpoint.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Spatial checkpoint SHA256 mismatch")
    parameters = np.frombuffer(raw, "<f4")
    if parameters.size != policy.buffers.parameter_words or not np.isfinite(parameters).all():
        raise ValueError("Spatial checkpoint layout differs or contains nonfinite parameters")
    weights = {}
    for name in ("input_kernel", "context_kernel", "action_kernel"):
        indices = getattr(model, name)
        weights[name] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    for name in ("local_weight", "local_bias", "context_weight", "context_bias"):
        weights[name] = parameters[getattr(model, name)[0]]
    for name in ("global_weight", "global_bias", "global_kernel", "readout_kernel", "output_weight", "output_bias"):
        weights[name] = parameters[getattr(model, name)]
    for i, (source, indices) in enumerate(model.priors):
        weights[f"prior_source_{i}"] = source
        weights[f"prior_weight_{i}"] = np.where(indices >= 0, parameters[np.maximum(indices, 0)], 0).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    for name, source in (("build.json", build), ("training.json", training), ("policy.bin", checkpoint)):
        (output / name).write_bytes(source.read_bytes())
    np.savez_compressed(output / "weights.npz", **weights)
    files = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
             for name in ("build.json", "training.json", "policy.bin", "weights.npz")}
    (output / "spatial-policy.json").write_text(json.dumps(dict(
        schema="puffer5-generals-spatial-v1", files=files, features=model.features,
        channels=model.channels,
        global_features=model.global_features, prior_count=len(model.priors),
        factory_source_sha256=hashlib.sha256(factory_source.read_bytes()).hexdigest(),
    ), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "factory-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    export_bundle(args.build, args.training, args.checkpoint, args.sha256, args.factory_source, args.output)


if __name__ == "__main__":
    main()

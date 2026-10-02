"""Export immutable spatial weights using the verified realized graph layout."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def export_bundle(build, training, checkpoint, sha256, factory_source, output, *,
                  serving_move_temperature=None, serving_split_temperature=None,
                  serving_early_route_temperature=None, serving_early_route_turns=None,
                  serving_neutral_route_bias=0.0, serving_weak_owned_route_penalty=0.0,
                  serving_doomed_attack_route_penalty=0.0):
    if (serving_move_temperature is None) != (serving_split_temperature is None):
        raise ValueError("Structured serving requires both action temperatures")
    if serving_move_temperature is not None and not all(
        np.isfinite(value) and value > 0 for value in (serving_move_temperature, serving_split_temperature)
    ):
        raise ValueError("Structured serving temperatures must be finite and positive")
    if (serving_early_route_temperature is None) != (serving_early_route_turns is None):
        raise ValueError("Early serving route temperature and turns must be paired")
    if serving_early_route_temperature is not None:
        from integrations.spatial_action_sampling import public_early_route_temperature

        if serving_move_temperature is None:
            raise ValueError("Early route schedule requires structured serving")
        public_early_route_temperature(np.zeros((1, 16 * 441), np.float32), serving_move_temperature,
                                       serving_early_route_temperature, serving_early_route_turns, np)
    if not np.isfinite(serving_neutral_route_bias) or serving_neutral_route_bias < 0 or (
            serving_neutral_route_bias and serving_move_temperature is None):
        raise ValueError("Neutral route bias requires structured serving and a finite nonnegative value")
    if not np.isfinite(serving_weak_owned_route_penalty) or serving_weak_owned_route_penalty < 0 or (
            serving_weak_owned_route_penalty and serving_move_temperature is None):
        raise ValueError("Weak owned route penalty requires structured serving and a finite nonnegative value")
    if not np.isfinite(serving_doomed_attack_route_penalty) or serving_doomed_attack_route_penalty < 0 or (
            serving_doomed_attack_route_penalty and serving_move_temperature is None):
        raise ValueError("Doomed attack route penalty requires structured serving and a finite nonnegative value")
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
    if serving_early_route_temperature is not None and (
            model.channels != 16 or manifest["config"]["python_environment"]["options"].get("public_scalar_ablation")):
        raise ValueError("Early route schedule requires full public scalar observations")
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
    serving_action_selection = (
        dict(mode="structured_sample", move_temperature=serving_move_temperature,
             split_temperature=serving_split_temperature)
        if serving_move_temperature is not None else dict(mode="argmax")
    )
    if serving_neutral_route_bias:
        serving_action_selection["neutral_route_bias"] = serving_neutral_route_bias
    if serving_early_route_temperature is not None:
        serving_action_selection["early_route_temperature"] = serving_early_route_temperature
        serving_action_selection["early_route_turns"] = serving_early_route_turns
    if serving_weak_owned_route_penalty:
        serving_action_selection["weak_owned_route_penalty"] = serving_weak_owned_route_penalty
    if serving_doomed_attack_route_penalty:
        serving_action_selection["doomed_attack_route_penalty"] = serving_doomed_attack_route_penalty
    (output / "spatial-policy.json").write_text(json.dumps(dict(
        schema="puffer5-generals-spatial-v1", files=files, features=model.features,
        channels=model.channels,
        global_features=model.global_features, prior_count=len(model.priors),
        factory_source_sha256=hashlib.sha256(factory_source.read_bytes()).hexdigest(),
        serving_action_selection=serving_action_selection,
    ), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    for name in ("build", "training", "checkpoint", "factory-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--serving-move-temperature", type=float)
    parser.add_argument("--serving-split-temperature", type=float)
    parser.add_argument("--serving-early-route-temperature", type=float)
    parser.add_argument("--serving-early-route-turns", type=int)
    parser.add_argument("--serving-neutral-route-bias", type=float, default=0.0)
    parser.add_argument("--serving-weak-owned-route-penalty", type=float, default=0.0)
    parser.add_argument("--serving-doomed-attack-route-penalty", type=float, default=0.0)
    args = parser.parse_args()
    export_bundle(args.build, args.training, args.checkpoint, args.sha256, args.factory_source, args.output,
                  serving_move_temperature=args.serving_move_temperature,
                  serving_split_temperature=args.serving_split_temperature,
                  serving_early_route_temperature=args.serving_early_route_temperature,
                  serving_early_route_turns=args.serving_early_route_turns,
                  serving_neutral_route_bias=args.serving_neutral_route_bias,
                  serving_weak_owned_route_penalty=args.serving_weak_owned_route_penalty,
                  serving_doomed_attack_route_penalty=args.serving_doomed_attack_route_penalty)


if __name__ == "__main__":
    main()

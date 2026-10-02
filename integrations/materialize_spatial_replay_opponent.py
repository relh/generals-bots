"""Write an exact native and portable bundle for a fitted replay opponent."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from integrations.export_spatial_policy_bundle import export_bundle
from integrations.spatial_policy_bundle import SpatialPlayerPolicy
from integrations.train_spatial_replay_opponent import TRAINABLE_NAMES


def materialize(source_bundle: Path, fit_dir: Path, factory_source: Path, output: Path) -> dict:
    from integrations.direct_spatial_optimization import DirectSpatial
    from integrations.memoryless_optimization import verify_configuration
    from metta_training.native_fabric import NativeFabricPolicy

    if output.exists():
        raise FileExistsError(output)
    record = json.loads((fit_dir / "fit.json").read_text())
    source_policy = SpatialPlayerPolicy(source_bundle)
    if record["source_policy_sha256"] != hashlib.sha256((source_bundle / "policy.bin").read_bytes()).hexdigest():
        raise ValueError("Fitted replay opponent source checkpoint differs")
    if record["weights_sha256"] != hashlib.sha256((fit_dir / "weights.npz").read_bytes()).hexdigest():
        raise ValueError("Fitted replay opponent weights checksum differs")
    if tuple(record["trainable_names"]) != TRAINABLE_NAMES:
        raise ValueError("Unexpected replay opponent trainable tensors")
    with np.load(fit_dir / "weights.npz") as archive:
        fitted = {name: archive[name] for name in archive.files}
    if fitted.keys() != source_policy.weights.keys():
        raise ValueError("Fitted replay opponent tensor names differ")
    for name, original in source_policy.weights.items():
        if fitted[name].shape != original.shape or fitted[name].dtype != original.dtype:
            raise ValueError(f"Fitted replay opponent tensor differs: {name}")
        if name not in TRAINABLE_NAMES and not np.array_equal(fitted[name], original):
            raise ValueError(f"Frozen replay opponent tensor changed: {name}")
        if not np.isfinite(fitted[name]).all():
            raise ValueError(f"Nonfinite replay opponent tensor: {name}")
    if record.get("mode") == "conditional_split":
        if (not np.array_equal(fitted["action_kernel"][:, :4], source_policy.weights["action_kernel"][:, :4])
                or not np.array_equal(fitted["readout_kernel"][:, :1764],
                                      source_policy.weights["readout_kernel"][:, :1764])
                or not np.array_equal(fitted["readout_kernel"][:, 3528:],
                                      source_policy.weights["readout_kernel"][:, 3528:])):
            raise ValueError("Conditional split fit altered a route or pass parameter")

    build = json.loads((source_bundle / "build.json").read_text())
    configuration = json.dumps(build["config"]["fabric"])
    verify_configuration(configuration)
    native = NativeFabricPolicy(configuration)
    model = DirectSpatial(native)
    parameters = np.frombuffer((source_bundle / "policy.bin").read_bytes(), "<f4").copy()
    if parameters.size != native.buffers.parameter_words:
        raise ValueError("Source native parameter layout differs")
    for name in TRAINABLE_NAMES:
        indices = np.asarray(getattr(model, name))
        values = fitted[name]
        if indices.shape != values.shape:
            raise ValueError(f"Native fitted tensor shape differs: {name}")
        valid = indices >= 0
        if np.any(values[~valid] != 0):
            raise ValueError(f"Fitted tensor uses absent native parameters: {name}")
        locations = indices[valid]
        if len(np.unique(locations)) != len(locations):
            raise ValueError(f"Native fitted tensor aliases parameters: {name}")
        parameters[locations] = values[valid]
    output.mkdir(parents=True)
    checkpoint = output / "policy.bin"
    checkpoint.write_bytes(parameters.astype("<f4", copy=False).tobytes())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    clone_metadata = {
        "schema": record["schema"], "source_policy_sha256": record["source_policy_sha256"],
        "dataset_manifest_sha256": record["dataset_manifest_sha256"],
        "dataset_opponent_policy_version_id": record["dataset_opponent_policy_version_id"],
        "selected_step": record["selected_step"],
        "selected_validation_nll": record["selected_validation_nll"],
        "mode": record.get("mode", "full_action"),
        "route_temperature": record["route_temperature"],
    }
    training_path = output / "training.json"
    # Keep the validated Puffer lineage record exact. The replay fit is
    # opponent-only metadata, recorded separately from the PPO run config.
    training_path.write_bytes((source_bundle / "training.json").read_bytes())
    export_bundle(source_bundle / "build.json", training_path, checkpoint, digest, factory_source,
                  output / "bundle", serving_move_temperature=record["route_temperature"],
                  serving_split_temperature=record["split_temperature"],
                  serving_route_half_weight=record.get("route_half_weight", 0.0),
                  serving_early_route_temperature=record.get("early_route_temperature"),
                  serving_early_route_turns=record.get("early_route_turns"),
                  serving_neutral_route_bias=source_policy.neutral_route_bias,
                  serving_weak_owned_route_penalty=source_policy.weak_owned_route_penalty,
                  serving_doomed_attack_route_penalty=source_policy.doomed_attack_route_penalty)
    manifest_path = output / "bundle" / "spatial-policy.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["replay_opponent_fit"] = clone_metadata
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    exported = SpatialPlayerPolicy(output / "bundle")
    maximum_error = max(float(np.max(np.abs(exported.weights[name] - fitted[name])))
                        for name in fitted if np.issubdtype(fitted[name].dtype, np.floating))
    if maximum_error != 0:
        raise ValueError(f"Native/portable replay opponent weights differ: {maximum_error}")
    if record.get("mode") == "conditional_split":
        if not np.array_equal(exported.weights["action_kernel"][:, :4], source_policy.weights["action_kernel"][:, :4]):
            raise ValueError("Export changed full-route local weights")
        if not np.array_equal(exported.weights["readout_kernel"][:, :1764], source_policy.weights["readout_kernel"][:, :1764]):
            raise ValueError("Export changed full-route global weights")
    result = {"schema": "coworld-classic-replay-opponent-bundle-v1", "checkpoint_sha256": digest,
              "fitted_weights_sha256": record["weights_sha256"], "max_weight_difference": maximum_error,
              "source_policy_sha256": record["source_policy_sha256"],
              "dataset_manifest_sha256": record["dataset_manifest_sha256"],
              "selected_step": record["selected_step"],
              "selected_validation_nll": record["selected_validation_nll"]}
    (output / "materialization.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--factory-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.source_bundle, args.fit_dir, args.factory_source, args.output), indent=2))


if __name__ == "__main__":
    main()

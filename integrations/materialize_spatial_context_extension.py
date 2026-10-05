"""Create an explicitly derived initialization, without rewriting its trained parent."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PARENT_MODEL = "811e8de55c3fe327a670a362b076f88fc8e56b9525a347ed89c945cfe9e8863b"
EXTENDED_MODEL = "fd64a611cb6ba1d721ddf519e5b15f7b26e207cd8c045d10483b04bbe2217f85"


def materialize(parent, build, output, factory_source, *, audit_native_layout=False):
    """Return a lossless initialization record, never a claim of additional training.

    Original training lineage is retained in initialization.json; migration.json
    records the parameter transformation. No completed.json is manufactured.
    The normal native learner-resume checks consume this derived snapshot.
    """
    from metta_training.learner import LearnerCheckpointIdentity, publish_checkpoint
    from metta_training.native_build import fabric_fingerprint

    from integrations.puffer_coworld_frozen_transfer import (
        BuildManifest,
        InitializationRecord,
        TrainingRecord,
        training_lineage_seeds,
    )
    from integrations.spatial_context_transfer import extend_flat, extend_learner, qualified_mapping
    from integrations.spatial_policy_bundle import SpatialPlayerPolicy

    parent, build, output = Path(parent), Path(build), Path(output)
    source_path = parent / "run/training.json"
    source = TrainingRecord.model_validate_json(source_path.read_text())
    target = BuildManifest.model_validate_json((build / "build.json").read_text())
    if (source.build.model_sha256 != PARENT_MODEL or target.model_sha256 != EXTENDED_MODEL
            or source.build.revision != target.revision
            or source.build.model_state_words != target.model_state_words):
        raise ValueError("Context migration requires the pinned parent and extended models")
    source_config = source.build.config.model_dump(exclude={"python_environment"})
    target_config = target.config.model_dump(exclude={"python_environment"})
    if source_config["fabric"]["options"]["context_radius"] != 1.01:
        raise ValueError("Context migration requires the original cross stencil")
    source_config["fabric"]["options"]["context_radius"] = 2.01
    if source_config != target_config:
        raise ValueError("Context migration changes configuration beyond the stencil")
    if hashlib.sha256((build / "puffer").read_bytes()).hexdigest() != target.binary_sha256:
        raise ValueError("Extended native executable differs from its build")
    # Source is deliberately imported from the frozen input, not the checkout's
    # potentially newer factory. Both model fingerprints cover that source.
    spec = importlib.util.spec_from_file_location("integrations.generals_fabric", factory_source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for manifest in (source.build, target):
        if fabric_fingerprint(manifest.config.fabric) != manifest.model_sha256:
            raise ValueError("Factory/framework differs from the pinned model fingerprint")
    parent_policy = SpatialPlayerPolicy(parent / "bundle")
    policy_bytes = (parent / "bundle/policy.bin").read_bytes()
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    checkpoint = parent / f"run/checkpoints/metta_generals/run/{source.config.total_timesteps:016d}.bin"
    if checkpoint.read_bytes() != policy_bytes:
        raise ValueError("Parent bundle differs from its final training checkpoint")
    learner_path = Path(str(checkpoint) + ".learner")
    learner_bytes = learner_path.read_bytes()
    identity = LearnerCheckpointIdentity.model_validate_json(Path(str(learner_path) + ".json").read_text())
    if (identity.policy_sha256 != policy_sha
            or identity.state_sha256 != hashlib.sha256(learner_bytes).hexdigest()
            or identity.run_sha256 != hashlib.sha256(source_path.read_bytes()).hexdigest()
            or identity.environment_sha256):
        raise ValueError("Parent policy, momentum, or original run identity differs")
    mapping = qualified_mapping()
    if audit_native_layout:
        import jax
        from metta_training.native_fabric import NativeFabricPolicy

        from integrations.direct_spatial_optimization import DirectSpatial
        from integrations.spatial_context_transfer import parameter_mapping
        from integrations.spatial_optimizer_layout import logical_optimizer_shapes

        models = []
        for manifest in (source.build, target):
            cpu = manifest.config.fabric.model_copy(update={"platform": "cpu"})
            # NativeFabricPolicy does not select a device from config.platform.
            with jax.default_device(jax.devices("cpu")[0]):
                policy = NativeFabricPolicy(cpu.model_dump_json())
                model = DirectSpatial(policy)
                logical_optimizer_shapes(model, policy.buffers, context_matrix=True)
            models.append((model, policy.buffers))
        (old, old_buffers), (new, new_buffers) = models
        np.testing.assert_array_equal(mapping, parameter_mapping(old, new, old_buffers, new_buffers))
    parameters = np.frombuffer(policy_bytes, "<f4")
    if parameters.size != 570668:
        raise ValueError("Parent checkpoint size differs from the actual native layout")
    extended = extend_flat(parameters, mapping).astype("<f4").tobytes()
    extended_learner = extend_learner(
        learner_bytes, mapping, source_count=parameters.size,
        expected_steps=source.config.total_timesteps,
        batch_steps=source.config.overrides["vec.total_agents"] * source.config.overrides["train.horizon"],
    )
    # This record is a derived initialization with zero new training steps.
    # Its own initialize reference points back to the untouched actual run.
    derived_config = source.config.model_dump(mode="json")
    derived_config["initialize"] = dict(run=str((parent / "run").resolve()),
                                        checkpoint=str(checkpoint.resolve()), sha256=policy_sha,
                                        restore_learner=True, allow_environment_transfer=True)
    record = TrainingRecord(build=target, config=derived_config)
    lineage = InitializationRecord(source=source, checkpoint_sha256=policy_sha,
                                   training_seeds=sorted(training_lineage_seeds(parent / "run", source)))
    output.mkdir(parents=True, exist_ok=False)
    run = output / "run"
    destination = run / checkpoint.relative_to(parent / "run")
    destination.parent.mkdir(parents=True)
    (run / "training.json").write_text(record.model_dump_json(indent=2) + "\n")
    (run / "initialization.json").write_text(lineage.model_dump_json(indent=2) + "\n")
    destination.write_bytes(extended)
    Path(str(destination) + ".learner").write_bytes(extended_learner)
    publish_checkpoint(str(destination), str(run / "training.json"), 0)

    bundle = output / "bundle"
    bundle.mkdir()
    (bundle / "build.json").write_text(target.model_dump_json(indent=2) + "\n")
    (bundle / "training.json").write_text(record.model_dump_json(indent=2) + "\n")
    (bundle / "policy.bin").write_bytes(extended)
    weights = {name: value.copy() for name, value in parent_policy.weights.items()}
    weights["context_kernel"] = np.pad(weights["context_kernel"], ((1, 1), (1, 1), (0, 0), (0, 0)))
    if audit_native_layout:
        values = np.frombuffer(extended, "<f4")
        for name in ("input_kernel", "context_kernel", "action_kernel"):
            indices = getattr(new, name)
            np.testing.assert_array_equal(weights[name], np.where(indices >= 0, values[np.maximum(indices, 0)], 0))
        for name in ("local_weight", "local_bias", "context_weight", "context_bias"):
            np.testing.assert_array_equal(weights[name], values[getattr(new, name)[0]])
        for name in ("global_weight", "global_bias", "global_kernel", "readout_kernel", "output_weight", "output_bias"):
            np.testing.assert_array_equal(weights[name], values[getattr(new, name)])
        for index, (prior_source, indices) in enumerate(new.priors):
            np.testing.assert_array_equal(weights[f"prior_source_{index}"], prior_source)
            np.testing.assert_array_equal(weights[f"prior_weight_{index}"],
                                          np.where(indices >= 0, values[np.maximum(indices, 0)], 0))
    np.savez_compressed(bundle / "weights.npz", **weights)
    metadata = json.loads((parent / "bundle/spatial-policy.json").read_text())
    metadata["files"] = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
                         for name in metadata["files"]}
    (bundle / "spatial-policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    SpatialPlayerPolicy(bundle)  # Check all exported metadata and tensor shapes.
    receipt = dict(schema=1, operation="zero_extend_context_1.01_to_2.01",
                   new_training_steps=0, source_run_sha256=identity.run_sha256,
                   source_policy_sha256=policy_sha, source_learner_sha256=identity.state_sha256,
                   source_model_sha256=PARENT_MODEL, target_model_sha256=EXTENDED_MODEL,
                   policy_sha256=hashlib.sha256(extended).hexdigest(),
                   learner_sha256=hashlib.sha256(extended_learner).hexdigest(),
                   mapping_sha256=hashlib.sha256(mapping.astype("<i4").tobytes()).hexdigest(),
                   source_parameter_count=parameters.size, target_parameter_count=len(mapping),
                   retained_momentum=True, retained_clocks=True, new_weights_and_momentum=0)
    (output / "migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return dict(run=str(run), checkpoint=str(destination), sha256=receipt["policy_sha256"],
                restore_learner=True, allow_environment_transfer=True)

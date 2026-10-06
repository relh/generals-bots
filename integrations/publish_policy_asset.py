"""Publish authentic completed PPO weights and optimizer state as a native asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrations.export_spatial_policy_bundle import realized_model
from integrations.learner_checkpoint import LearnerCheckpoint
from integrations.native_spatial_asset import canonical_json, load_asset, training_contract, write_asset


def completed_checkpoint(run, record, checkpoint, expected_sha, parameter_count):
    """Validate the final per-GPU clock and hashes actually published by Puffer."""
    from metta_training.learner import LearnerCheckpointIdentity

    from integrations.puffer_coworld_frozen_transfer import TrainingResult

    completed = TrainingResult.model_validate_json((run / "completed.json").read_text())
    relative = checkpoint.resolve().relative_to(run.resolve())
    batch = record.config.overrides["vec.total_agents"] * record.config.overrides["train.horizon"]
    if batch <= 0:
        raise ValueError("Completed run must have positive rollout geometry")
    if record.config.overrides.get("train.gpus", 1) != 1:
        raise ValueError("A native learner asset requires an actual single-GPU checkpoint")
    # total_timesteps is the absolute target clock, including restored history;
    # Puffer completes whole rollout batches and publishes the rounded clock.
    expected_step = record.config.total_timesteps // batch * batch
    if (
        relative not in completed.checkpoints
        or relative != completed.final_checkpoint
        or completed.trained_timesteps != expected_step
        or completed.revision != record.build.revision
    ):
        raise ValueError("PPO asset must be the actual completed final training checkpoint")
    policy_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if policy_sha != expected_sha:
        raise ValueError("Completed checkpoint digest differs")
    snapshot = Path(str(checkpoint) + ".learner")
    learner = LearnerCheckpoint.read(snapshot, parameter_count)
    if (
        learner.agent_steps != completed.trained_timesteps
        or learner.epoch * batch != learner.agent_steps
        or int(checkpoint.stem) != learner.agent_steps
    ):
        raise ValueError("Completed optimizer clock differs from the actual PPO budget")
    identity_path = Path(str(snapshot) + ".json")
    identity = LearnerCheckpointIdentity.model_validate_json(identity_path.read_text())
    if (
        identity.policy_sha256 != policy_sha
        or identity.state_sha256 != hashlib.sha256(snapshot.read_bytes()).hexdigest()
        or identity.run_sha256 != hashlib.sha256((run / "training.json").read_bytes()).hexdigest()
        or identity.environment_sha256
    ):
        raise ValueError("Published checkpoint identity differs from its current device-resident run")
    return learner, snapshot, completed, identity_path


def curriculum_lineage(curriculum, options):
    """Authenticate the exact source-only pool before adding its actual seeds."""
    data = json.loads(curriculum.read_text())
    pool = Path(options["coworld_position_pool"])
    if (data.get("schema") != "classic-midgame-positions-v1"
            or data.get("split") != "source" or data.get("teacher_labels_enabled") is not False
            or pool.resolve() != curriculum.with_name("positions.npz").resolve()
            or data["positions_sha256"] != options["coworld_position_pool_sha256"]
            or hashlib.sha256(pool.read_bytes()).hexdigest() != data["positions_sha256"]):
        raise ValueError("Published curriculum differs from the authentic source-only training pool")
    actual_seeds = [data["root_training_seed"], *data["map_seeds"]]
    if any(type(seed) is not int or seed < 0 for seed in actual_seeds):
        raise ValueError("Curriculum requires actual nonnegative map and root seeds")
    return set(actual_seeds), hashlib.sha256(curriculum.read_bytes()).hexdigest()


def publish(build_path, training_path, checkpoint, expected_sha, sampler, factory_source, output, *, curriculum=None):
    from integrations.puffer_coworld_frozen_transfer import BuildManifest, InitializationRecord, TrainingRecord

    build = BuildManifest.model_validate_json(build_path.read_text())
    record = TrainingRecord.model_validate_json(training_path.read_text())
    if record.build != build:
        raise ValueError("Completed training and compiled model identities differ")
    run = training_path.parent
    config = build.config.fabric.model_dump(mode="json")
    source_sha = hashlib.sha256(factory_source.read_bytes()).hexdigest()
    native, _, model_sha, abi_sha = realized_model(canonical_json(config).decode(), factory_source, source_sha)
    if model_sha != build.model_sha256:
        raise ValueError("Completed checkpoint does not bind the current compiled model")
    learner, snapshot, completed, identity_path = completed_checkpoint(
        run, record, checkpoint, expected_sha, native.buffers.parameter_words
    )
    overrides = record.config.overrides
    seeds = {record.config.seed}
    ancestors = dict(
        training_record=hashlib.sha256(training_path.read_bytes()).hexdigest(),
        compiled_build=hashlib.sha256(build_path.read_bytes()).hexdigest(),
        completed_run=hashlib.sha256((run / "completed.json").read_bytes()).hexdigest(),
        checkpoint_identity=hashlib.sha256(identity_path.read_bytes()).hexdigest(),
    )
    if curriculum is not None:
        actual_seeds, manifest_digest = curriculum_lineage(curriculum, build.config.python_environment.options)
        seeds.update(actual_seeds)
        ancestors["training_curriculum"] = manifest_digest
    start = 0
    if record.config.initialize:
        initialization = record.config.initialize
        asset = load_asset(initialization.asset, manifest_sha256=initialization.manifest_sha256)
        asset.verify_target(factory_source_sha256=source_sha, model_sha256=model_sha, abi_sha256=abi_sha)
        initialized = InitializationRecord.model_validate_json((run / "initialization.json").read_text())
        if (
            initialized.asset.resolve() != initialization.asset.resolve()
            or initialized.manifest_sha256 != initialization.manifest_sha256
            or initialized.checkpoint_sha256 != asset.metadata["policy_sha256"]
            or initialized.restore_learner != initialization.restore_learner
            or initialized.training_seeds != asset.metadata["training_seeds"]
            or hashlib.sha256((run / "initial-policy.bin").read_bytes()).hexdigest() != initialized.checkpoint_sha256
        ):
            raise ValueError("Completed run initialization differs from its authenticated source asset")
        seeds.update(asset.metadata["training_seeds"])
        ancestors["initial_asset"] = initialization.manifest_sha256
        if initialization.restore_learner:
            if asset.learner is None:
                raise ValueError("Restored run source asset has no authentic learner")
            if asset.metadata["learner_configuration"] != dict(seed=record.config.seed, overrides=overrides):
                raise ValueError("Restored source learner configuration differs from the completed run")
            if asset.metadata["training_contract"] != training_contract(
                build.config.python_environment.options, overrides
            ):
                raise ValueError("Restored source training objective differs from the completed run")
            if (run / "initial-policy.bin.learner").read_bytes() != asset.learner:
                raise ValueError("Completed run restored learner differs from its authenticated source")
            start = LearnerCheckpoint.from_bytes(asset.learner, asset.metadata["parameter_count"]).agent_steps
    if learner.agent_steps <= start:
        raise ValueError("PPO publication requires actual new environment steps")
    provenance = dict(
        operation="reinforcement_learning",
        ancestors=ancestors,
        reinforcement_learning_steps_added=learner.agent_steps - start,
    )
    path = write_asset(
        output,
        fabric=config,
        factory_source_sha256=source_sha,
        model_sha256=model_sha,
        abi_sha256=abi_sha,
        policy=checkpoint,
        sampler=sampler,
        provenance=provenance,
        training_seeds=sorted(seeds),
        learner=snapshot,
        learner_configuration=dict(seed=record.config.seed, overrides=overrides),
        training_contract=training_contract(build.config.python_environment.options, overrides),
    )
    return dict(
        asset=str(path),
        manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        environment_steps_added=learner.agent_steps - start,
        policy_sha256=expected_sha,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "training", "checkpoint", "sampler", "factory-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            publish(
                args.build,
                args.training,
                args.checkpoint,
                args.sha256,
                json.loads(args.sampler.read_text()),
                args.factory_source,
                args.output,
            )
        )
    )


if __name__ == "__main__":
    main()

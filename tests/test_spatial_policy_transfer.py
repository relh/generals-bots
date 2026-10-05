"""Current model transfer must preserve ABI, provenance and optimizer semantics."""
import hashlib
import os
import struct

import pytest

pytest.importorskip("metta_training")

from integrations.launch_spatial_selfplay_training import load_pinned_trainer


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setenv("METTA_SPATIAL_MUON_DENSE_ORIENTATION", "storage")
    monkeypatch.setenv("METTA_SPATIAL_MUON_CONTEXT_MATRIX", "0")
    return load_pinned_trainer()


@pytest.fixture
def source(trainer):
    build = trainer.BuildManifest.model_validate({
        "binary_sha256": hashlib.sha256(b"test binary").hexdigest(),
        "model_sha256": "a" * 64, "model_state_words": 77684,
        "environment_sha256": "b" * 64,
        "config": {"environment": "metta_generals", "environment_backend": "cuda",
            "fabric": {"factory": trainer.SPATIAL_FACTORY, "observation_size": 7056,
                       "action_sizes": [3529], "options": {"channels": 16}, "platform": "cuda"},
            "python_environment": {"factory": "integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment",
                "device_resident": True, "workers": "inline",
                "spec": {"agents": 8192, "observation_size": 7056, "action_sizes": [3529]},
                "options": {"coworld_classic": True, "horizon": 2000, "parallel_games": 8192,
                    "shaping_gamma": .999, "terminal_reward_mode": "win_only", "balance_opponent_sides": True}}},
    })
    run = trainer.RunConfig(total_timesteps=8388608, seed=8842, overrides={
        "train.gamma": .999, "train.horizon": 256, "train.minibatch_size": 8192, "vec.total_agents": 8192})
    return trainer.TrainingRecord(build=build, config=run)


def test_distribution_changes_preserve_abi_but_reward_drift_rejected(trainer, source):
    target = source.build.model_copy(deep=True)
    target.config.python_environment.options["opponent_weights"] = [1, 2, 3]
    trainer.validate_spatial_transfer(source, target, source.config)
    target.config.python_environment.options["reward_scale"] = .25
    with pytest.raises(ValueError, match="reward semantics"):
        trainer.validate_spatial_transfer(source, target, source.config)


@pytest.mark.parametrize("change", ["model", "state", "factory", "codec", "rules"])
def test_incompatible_transfer_rejected(trainer, source, change):
    target = source.build.model_copy(deep=True)
    if change == "model": target = target.model_copy(update={"model_sha256": "c" * 64})
    if change == "state": target = target.model_copy(update={"model_state_words": target.model_state_words + 1})
    if change == "factory": target.config.fabric.factory = "deleted:old_policy"
    if change == "codec":
        env = target.config.python_environment
        wrong_env = env.model_copy(update={"spec": env.spec.model_copy(update={"action_sizes": [1765, 2]})})
        target = target.model_copy(update={"config": target.config.model_copy(update={"python_environment": wrong_env})})
    if change == "rules": target.config.python_environment.options["coworld_classic"] = False
    with pytest.raises(ValueError):
        trainer.validate_spatial_transfer(source, target, source.config)


@pytest.mark.parametrize("restore", ["restore_learner", "restore_ema", "restore_horde", "restore_rnd", "migrate_classic_rollout"])
def test_policy_only_never_restores_optimizer_or_auxiliary_state(trainer, restore):
    with pytest.raises(ValueError, match="fresh optimizer"):
        trainer.CheckpointInitialization(run="source", checkpoint="source/checkpoints/policy.bin",
            sha256="a" * 64, allow_environment_transfer=True, allow_policy_only_transfer=True, **{restore: True})


def warmstart(trainer, source, data):
    return trainer.SupervisedPolicyTransfer(source=source,
        source_training_sha256=trainer.training_record_sha256(source), source_checkpoint_sha256="c" * 64,
        checkpoint_sha256=hashlib.sha256(data).hexdigest(), parameter_count=len(data)//4,
        training_seeds=[8842], optimizer_updates=4, training_data_sha256="d" * 64)


def test_supervised_artifact_retains_real_lineage_without_rl_clocks(trainer, source):
    artifact = warmstart(trainer, source, struct.pack("<4f", 1, 2, 3, 4))
    assert artifact.optimizer_updates == 4
    assert artifact.source.config.total_timesteps == source.config.total_timesteps
    altered = artifact.model_dump(mode="json")
    altered["source"]["config"]["seed"] = 12
    with pytest.raises(ValueError, match="training identity"):
        trainer.SupervisedPolicyTransfer.model_validate(altered)


def test_actual_native_prepare_accepts_verified_supervised_policy_with_fresh_state(trainer, source, tmp_path, monkeypatch):
    # Exercise preparation itself, including all checkpoint checks. CUDA build
    # hooks are independently checked by launcher preflight tests.
    build = tmp_path / "build"
    ini = build / "source/config"
    ini.mkdir(parents=True)
    (build / "puffer").write_bytes(b"test binary")
    (build / "build.json").write_text(source.build.model_dump_json())
    (ini / "default.ini").write_text("[base]\ncudagraphs = -1\n[vec]\ntotal_agents = 8192\nnum_buffers = 1\n[train]\nhorizon = 256\nminibatch_size = 8192\ngamma = .999\ngpus = 1\n")
    (ini / "metta_generals.ini").write_text("")
    artifact_dir = tmp_path / "warmstart"
    checkpoint = artifact_dir / "checkpoints/policy.bin"
    checkpoint.parent.mkdir(parents=True)
    data = struct.pack("<4f", 1, 2, 3, 4)
    checkpoint.write_bytes(data)
    artifact = warmstart(trainer, source, data)
    (artifact_dir / "policy-transfer.json").write_text(artifact.model_dump_json())
    monkeypatch.setattr(trainer, "fabric_runtime", lambda *a: ({}, dict(os.environ)))
    monkeypatch.setattr(trainer, "cuda_runtime_environment", lambda e: e)
    run = source.config.model_copy(update={"initialize": trainer.CheckpointInitialization(
        run=artifact_dir, checkpoint=checkpoint, sha256=artifact.checkpoint_sha256,
        allow_environment_transfer=True, allow_policy_only_transfer=True)})
    prepared = trainer.prepare_run(build, tmp_path / "fresh-run", run)
    assert prepared.initial_parameters == data
    assert prepared.initial_learner == b""
    assert prepared.initialization.supervised_transfer == artifact
    assert prepared.environment["METTA_INITIAL_LEARNER"] == ""
    # Same declared artifact, changed policy bytes must fail before launch.
    checkpoint.write_bytes(struct.pack("<4f", 1, 2, 3, 5))
    with pytest.raises(ValueError, match="checkpoint digest"):
        trainer.prepare_run(build, tmp_path / "invalid-run", run)


def test_owned_lineage_reads_supervised_artifact_and_detects_wrong_bundle(trainer, source, tmp_path):
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    (run / "checkpoints").mkdir(parents=True)
    data = struct.pack("<4f", 1, 2, 3, 4)
    artifact = warmstart(trainer, source, data).model_copy(update={"training_seeds": [8842, 12345]})
    (bundle / "training.json").write_text(source.model_dump_json())
    (bundle / "build.json").write_text(source.build.model_dump_json())
    (bundle / "policy.bin").write_bytes(data)
    (run / "checkpoints/supervised.bin").write_bytes(data)
    (run / "policy-transfer.json").write_text(artifact.model_dump_json())
    assert trainer.policy_training_lineage_seeds(bundle, run) == {8842, 12345}
    assert not (run / "training.json").exists()
    (bundle / "policy.bin").write_bytes(struct.pack("<4f", 1, 2, 3, 5))
    with pytest.raises(ValueError, match="checkpoint ABI"):
        trainer.policy_training_lineage_seeds(bundle, run)

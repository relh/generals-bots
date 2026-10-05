"""Regressions for the launch binding missed by the former CPU image proof."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations import launch_spatial_selfplay_training as launcher
from integrations import portable_classic_pilot as pilot
from integrations.entropy_resume import entropy_resume_source


def test_current_trainer_binding_and_mutation_rejection(tmp_path):
    source = launcher.verify_trainer_source()
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n# unreviewed edit\n")
    with pytest.raises(ValueError, match="Pinned Puffer trainer changed"):
        launcher.verify_trainer_source(changed)


def test_entropy_resume_transform_preserves_migration_and_seed_guards():
    source = launcher.verify_trainer_source().read_text()
    patched = entropy_resume_source(source)
    compile(patched, "pinned-trainer", "exec")
    assert "source.config.seed != config.seed" in patched
    assert "not reference.migrate_classic_rollout" in patched
    assert "and not entropy_resume_overrides_compatible" in patched
    with pytest.raises(ValueError, match="Pinned learner resume guard differs"):
        entropy_resume_source(patched)


def test_portable_cpu_proof_invokes_real_launcher_and_bootstrap(tmp_path):
    with patch.object(pilot.subprocess, "run") as run:
        pilot.cpu_preflight(tmp_path / "build", tmp_path / "config.json", tmp_path / "prepared")
    argv = run.call_args.args[0]
    env = run.call_args.kwargs["env"]
    assert argv[1:4] == ["-m", "integrations.launch_spatial_selfplay_training", "preflight"]
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["METTA_MEMORYLESS_OPTIMIZATION"] == "1"
    assert env["PYTHONPATH"].split(":")[0].endswith("integrations/puffer_bootstrap")
    assert run.call_args.kwargs["check"]


@pytest.mark.parametrize("entropy", ["0", "1"])
def test_real_imported_launcher_retains_checkpoint_abi_and_native_guard(tmp_path, entropy):
    pytest.importorskip("metta_training")
    with patch.dict(os.environ, {
        "METTA_SPATIAL_MUON_DENSE_ORIENTATION": "canonical",
        "METTA_SPATIAL_MUON_CONTEXT_MATRIX": "0",
        "METTA_ALLOW_ENTROPY_COEFFICIENT_RESUME": entropy,
    }):
        trainer = launcher.load_pinned_trainer()
    assert trainer.__name__ == "metta_training.puffer"
    assert trainer.PUFFER_REVISION == "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
    assert "migrate_classic_rollout" in trainer.CheckpointInitialization.model_fields
    (tmp_path / "puffer").write_bytes(b"unqualified native build")
    with pytest.raises(ValueError, match="orientation does not match"):
        trainer.prepare_run(tmp_path, tmp_path / "out", None)

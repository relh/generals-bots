"""Regressions for the launch binding missed by the former CPU image proof."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations import launch_spatial_selfplay_training as launcher


def test_current_trainer_binding_and_mutation_rejection(tmp_path):
    source = launcher.verify_trainer_source()
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n# unreviewed edit\n")
    with pytest.raises(ValueError, match="Pinned Puffer trainer changed"):
        launcher.verify_trainer_source(changed)




def test_real_imported_launcher_retains_checkpoint_abi_and_native_guard(tmp_path):
    pytest.importorskip("metta_training")
    with patch.dict(os.environ, {
        "METTA_SPATIAL_MUON_DENSE_ORIENTATION": "canonical",
        "METTA_SPATIAL_MUON_CONTEXT_MATRIX": "0",
    }):
        trainer = launcher.load_pinned_trainer()
    assert trainer.__name__ == "metta_training.puffer"
    assert trainer.PUFFER_REVISION == "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
    assert "migrate_classic_rollout" in trainer.CheckpointInitialization.model_fields
    (tmp_path / "puffer").write_bytes(b"unqualified native build")
    with pytest.raises(ValueError, match="orientation does not match"):
        trainer.prepare_run(tmp_path, tmp_path / "out", None)

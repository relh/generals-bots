"""Native initialization accepts one explicit asset and honest learner behavior."""
import json

import pytest
pytest.importorskip('metta_training')

from integrations.launch_spatial_selfplay_training import load_pinned_trainer


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setenv('METTA_SPATIAL_MUON_DENSE_ORIENTATION','storage')
    monkeypatch.setenv('METTA_SPATIAL_MUON_CONTEXT_MATRIX','0')
    return load_pinned_trainer()


def test_initializer_requires_explicit_learner_choice_and_rejects_source_records(trainer):
    current=dict(asset='asset.json',manifest_sha256='a'*64,restore_learner=False)
    assert trainer.CheckpointInitialization(**current).restore_learner is False
    with pytest.raises(ValueError):
        trainer.CheckpointInitialization(asset='asset.json',manifest_sha256='a'*64)
    for field,value in [('run','source'),('checkpoint','old.bin'),('sha256','b'*64),
                        ('allow_environment_transfer',True),('allow_policy_only_transfer',True),
                        ('migrate_classic_rollout',False)]:
        with pytest.raises(ValueError):
            trainer.CheckpointInitialization(**current,**{field:value})


def test_initializer_requires_complete_manifest_identity(trainer):
    with pytest.raises(ValueError):
        trainer.CheckpointInitialization(asset='asset.json',manifest_sha256='abc',restore_learner=True)


def test_actual_run_lineage_preserves_asset_seeds_without_loading_old_records(trainer,tmp_path):
    record=type('ActualRun',(),{'config':type('Config',(),{'seed':8857,'initialize':True})()})()
    (tmp_path/'initialization.json').write_text(json.dumps(dict(asset='/assets/cold/asset.json',
        manifest_sha256='a'*64,checkpoint_sha256='b'*64,training_seeds=[7793,8842],restore_learner=False)))
    assert trainer.training_lineage_seeds(tmp_path,record)=={7793,8842,8857}
    bad=json.loads((tmp_path/'initialization.json').read_text());bad['source']={'config':{'seed':1}}
    (tmp_path/'initialization.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        trainer.training_lineage_seeds(tmp_path,record)

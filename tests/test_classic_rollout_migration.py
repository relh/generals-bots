import struct

import pytest

from integrations.classic_rollout_migration import migrate_learner
from integrations.learner_checkpoint import LEARNER_HEADER, LearnerCheckpoint


def fixture():
    source = {'vec.total_agents': 8192, 'train.horizon': 256,
              'train.minibatch_size': 8192, 'train.anneal_lr': 0,
              'train.ent_coef': 0.0, 'vec.num_buffers': 1,
              'train.replay_ratio': 0.5, 'train.learning_rate': .0002}
    target = dict(source, **{'vec.total_agents': 2048})
    data = LEARNER_HEADER.pack(b'METTAL01', 1192, 2499805184, 2, .0002) + struct.pack('<ff', 3, -7)
    return source, target, data


def test_migration_changes_only_epoch_and_preserves_optimizer_bytes():
    source, target, data = fixture()
    migrated, receipt = migrate_learner(data, 2, source, target)
    state = LearnerCheckpoint.from_bytes(migrated, 2)
    assert state.epoch == 4768
    assert state.agent_steps == 2499805184
    assert state.epoch * 2048 * 256 == state.agent_steps
    assert data[:8] == migrated[:8] and data[16:] == migrated[16:]
    assert receipt['momentum_preserved_exactly']
    assert receipt['source_sha256'] != receipt['target_sha256']


@pytest.mark.parametrize('key,value', [
    ('train.horizon', 64), ('train.minibatch_size', 2048),
    ('train.learning_rate', .001), ('train.ent_coef', .01),
    ('vec.total_agents', 4096),
])
def test_unrelated_treatment_changes_rejected(key, value):
    source, target, data = fixture()
    target[key] = value
    with pytest.raises(ValueError):
        migrate_learner(data, 2, source, target)


def test_schedule_and_corrupt_clock_rejected():
    source, target, data = fixture()
    for key, value in [('train.anneal_lr', 1), ('train.ent_coef', .01)]:
        with pytest.raises(ValueError, match='constant-rate'):
            migrate_learner(data, 2, dict(source, **{key: value}), dict(target, **{key: value}))
    corrupt = bytearray(data)
    struct.pack_into('<Q', corrupt, 8, 1191)
    with pytest.raises(ValueError, match='clock'):
        migrate_learner(bytes(corrupt), 2, source, target)
    with pytest.raises(ValueError, match='payload size'):
        migrate_learner(data[:-1], 2, source, target)

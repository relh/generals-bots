"""Portable METTAL01 checks must reject corrupted optimizer state."""
import dataclasses
import struct

import pytest

from integrations.learner_checkpoint import LearnerCheckpoint


def snapshot(*, magic=b'METTAL01', count=2, rate=.0002, momentum=(1., -2.)):
    return struct.pack('<8sQQQf', magic, 3, 6291456, count, rate) + struct.pack('<' + 'f' * len(momentum), *momentum)


def test_preserves_exact_counters_and_float32_rate(tmp_path):
    path = tmp_path / 'checkpoint.learner'
    data = snapshot()
    path.write_bytes(data)
    state = LearnerCheckpoint.read(path, 2)
    assert dataclasses.asdict(state) == dict(epoch=3, agent_steps=6291456,
                                          parameter_count=2,
                                          learning_rate=struct.unpack('<f', struct.pack('<f', .0002))[0])
    assert path.read_bytes() == data


@pytest.mark.parametrize('data,count', [
    (b'METTAL01', 2),
    (snapshot(magic=b'UNKNOWN!'), 2),
    (snapshot(), 3),
    (snapshot()[:-1], 2),
    (snapshot() + b'\0', 2),
    (snapshot(count=0, momentum=()), 0),
    (snapshot(rate=-1), 2),
    (snapshot(rate=float('nan')), 2),
    (snapshot(rate=float('inf')), 2),
    (snapshot(momentum=(float('nan'), 0)), 2),
    (snapshot(momentum=(0, float('inf'))), 2),
])
def test_rejects_invalid_state_without_modifying_it(tmp_path, data, count):
    path = tmp_path / 'checkpoint.learner'
    path.write_bytes(data)
    with pytest.raises(ValueError):
        LearnerCheckpoint.read(path, count)
    assert path.read_bytes() == data


def test_matches_optional_pinned_trainer_reader(tmp_path):
    upstream = pytest.importorskip('metta_training.learner').LearnerCheckpoint
    path = tmp_path / 'checkpoint.learner'
    path.write_bytes(snapshot())
    assert dataclasses.asdict(LearnerCheckpoint.read(path, 2)) == upstream.read(path, 2).model_dump()

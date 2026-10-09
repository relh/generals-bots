"""Actual snapshot identity and rounded absolute clocks gate asset publication."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from integrations.learner_checkpoint import LEARNER_HEADER
from integrations.publish_policy_asset import completed_checkpoint


def fixture(tmp_path, *, budget=35, steps=32):
    run = tmp_path / 'run'
    checkpoint = run / 'checkpoints' / f'{steps:016d}.bin'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(bytes(16))
    (run / 'training.json').write_text('{"actual":"record"}')
    state = Path(str(checkpoint) + '.learner')
    state.write_bytes(LEARNER_HEADER.pack(b'METTAL01', steps // 8, steps, 4, .0002) + bytes(16))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(state) + '.json').write_text(json.dumps(dict(policy_sha256=sha(checkpoint), state_sha256=sha(state),
                                                        run_sha256=sha(run / 'training.json'), environment_sha256=[])))
    revision = '6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2'
    (run / 'completed.json').write_text(json.dumps(dict(checkpoints=[str(checkpoint.relative_to(run))],
         final_checkpoint=str(checkpoint.relative_to(run)), revision=revision, trained_timesteps=steps, metrics='metrics.ini')))
    record = SimpleNamespace(config=SimpleNamespace(total_timesteps=budget,
              overrides={'vec.total_agents': 2, 'train.horizon': 4}), build=SimpleNamespace(revision=revision))
    return run, record, checkpoint, sha(checkpoint)


def test_completed_clock_is_absolute_and_rounds_to_rollout(tmp_path):
    # A run restored at16 and targeted35 finishes32, adding16 real steps.
    args = fixture(tmp_path)
    learner, _, completed, _ = completed_checkpoint(*args, 4)
    assert learner.agent_steps == completed.trained_timesteps == 32
    assert learner.agent_steps - 16 == 16


def test_replaced_learner_with_same_clock_is_rejected(tmp_path):
    args = fixture(tmp_path)
    path = Path(str(args[2]) + '.learner')
    data = bytearray(path.read_bytes())
    data[-4:] = b'\x00\x00\x80\x3f'
    path.write_bytes(data)
    with pytest.raises(ValueError, match='checkpoint identity'):
        completed_checkpoint(*args, 4)


def test_early_checkpoint_or_changed_training_record_is_rejected(tmp_path):
    args = fixture(tmp_path, steps=24)
    with pytest.raises(ValueError, match='actual completed final'):
        completed_checkpoint(*args, 4)
    args = fixture(tmp_path / 'other')
    (args[0] / 'training.json').write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='checkpoint identity'):
        completed_checkpoint(*args, 4)

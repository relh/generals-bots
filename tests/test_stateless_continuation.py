"""Fail closed before qualification; authenticate cumulative clocks independently."""
import json
import struct
from pathlib import Path
import pytest
from integrations import stateless_continuation_run as runner


def test_pending_plan_cannot_prepare(tmp_path):
    with pytest.raises(ValueError, match='pending'):
        runner.prepare(tmp_path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_sealed_plan_requires_each_hash(tmp_path, monkeypatch):
    plan = runner.read(runner.PLAN_PATH)
    plan.update(status='sealed after qualifying terminal audit', qualification_job_id='job-test1')
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'PLAN_PATH', path)
    with pytest.raises(ValueError, match='Unbound'):
        runner.bound_plan(tmp_path)


@pytest.mark.parametrize('steps,epoch', [(4194304, 8), (33554432, 64)])
def test_exact_learner_clock(tmp_path, steps, epoch):
    path = tmp_path / 'state'
    path.write_bytes(struct.pack('<8sQQQf', b'METTAL01', epoch, steps, 1, .0002) + struct.pack('<f', .1))
    runner.clock(path, 1, steps, epoch)
    with pytest.raises(ValueError, match='clock'):
        runner.clock(path, 1, steps + 1, epoch)
    with pytest.raises(ValueError, match='clock'):
        runner.clock(path, 1, steps, epoch + 1)

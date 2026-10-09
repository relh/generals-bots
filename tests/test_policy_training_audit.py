"""Fail closed before qualification; authenticate cumulative clocks independently."""
import json
import struct
from pathlib import Path
import pytest
from integrations import policy_training_audit as runner






@pytest.mark.parametrize('steps,epoch', [(4194304, 8), (33554432, 64)])
def test_exact_learner_clock(tmp_path, steps, epoch):
    path = tmp_path / 'state'
    path.write_bytes(struct.pack('<8sQQQf', b'METTAL01', epoch, steps, 1, .0002) + struct.pack('<f', .1))
    runner.clock(path, 1, steps, epoch)
    with pytest.raises(ValueError, match='clock'):
        runner.clock(path, 1, steps + 1, epoch)
    with pytest.raises(ValueError, match='clock'):
        runner.clock(path, 1, steps, epoch + 1)




@pytest.mark.parametrize('failure', ['exit', 'timeout'])
def test_entrypoint_retains_partial_evidence(tmp_path, failure):
    import os
    import subprocess
    import sys
    import tarfile
    from integrations.bounded_policy_entrypoint import run
    results, output = tmp_path / 'results', tmp_path / 'output'
    body = "from pathlib import Path; import time; p=Path(" + repr(str(results)) + "); p.mkdir(); (p/'partial.bin').write_bytes(b'evidence'); "
    body += 'raise SystemExit(1)' if failure == 'exit' else 'time.sleep(30)'
    expected = RuntimeError if failure == 'exit' else subprocess.TimeoutExpired
    with pytest.raises(expected):
        run([([sys.executable, '-c', body], dict(os.environ))], results, output,
            5 if failure == 'exit' else .25, kind='fixture', completion=lambda _: {})
    receipt = json.loads((output / 'collection.json').read_text())
    assert receipt['complete'] is False and 'partial.bin' in receipt['files']
    assert (output / 'FAILED.json').is_file() and not (output / 'COMPLETED.json').exists()
    with tarfile.open(output / 'results.tar.gz') as archive:
        assert archive.extractfile('generals/partial.bin').read() == b'evidence'


def test_entrypoint_publishes_success_after_collection(tmp_path):
    import os
    import sys
    from integrations.bounded_policy_entrypoint import run
    results, output = tmp_path / 'results', tmp_path / 'output'
    results.mkdir(); (results / 'plan.json').write_text('{}\n')
    run([([sys.executable, '-c', 'pass'], dict(os.environ))], results, output, 5,
        kind='fixture', completion=lambda _: {'selected': False})
    marker = json.loads((output / 'COMPLETED.json').read_text())
    assert marker['selected'] is False
    assert marker['results_sha256'] == runner.digest(output / 'results.tar.gz')
    assert json.loads((output / 'collection.json').read_text())['complete'] is True


def test_cleanup_wait_failure_does_not_skip_collection(tmp_path, monkeypatch):
    import subprocess
    from integrations import bounded_policy_entrypoint as entry
    results, output = tmp_path / 'results', tmp_path / 'output'
    results.mkdir(); (results / 'partial.bin').write_bytes(b'partial')
    class Child:
        pid = 12345
        def wait(self, timeout):
            raise subprocess.TimeoutExpired('fixture', timeout)
    monkeypatch.setattr(entry.subprocess, 'Popen', lambda *a, **k: Child())
    def gone(*_):
        raise ProcessLookupError()
    monkeypatch.setattr(entry.os, 'killpg', gone)
    with pytest.raises(subprocess.TimeoutExpired) as error:
        entry.run([(['fixture'], {})], results, output, 1, kind='fixture', completion=lambda _: {})
    assert error.value.timeout <= 1  # Original deadline exception, not cleanup's ten-second wait.
    failure = json.loads((output / 'FAILED.json').read_text())
    assert 'cleanup_error' in failure
    assert (output / 'results.tar.gz').is_file()
    assert json.loads((output / 'collection.json').read_text())['complete'] is False

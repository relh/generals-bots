import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from integrations import autoresearch_gpu
from integrations.autoresearch_classic_job import run_phase


def test_phase_success_and_failure_preserve_logs(tmp_path):
    run_phase([sys.executable, '-c', 'print("result")'], tmp_path / 'ok', 5, os.environ.copy())
    assert (tmp_path / 'ok').read_text().strip() == 'result'
    with pytest.raises(RuntimeError, match='exit 6'):
        run_phase([sys.executable, '-c', 'print("evidence"); exit(6)'],
                  tmp_path / 'failure', 5, os.environ.copy())
    assert (tmp_path / 'failure').read_text().strip() == 'evidence'


def test_timeout_waits_for_resistant_owned_process(tmp_path):
    pid = tmp_path / 'pid'
    code = (f'from pathlib import Path; import os,signal,time; '
            f'Path({str(pid)!r}).write_text(str(os.getpid())); '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)')
    with pytest.raises(subprocess.TimeoutExpired):
        run_phase([sys.executable, '-c', code], tmp_path / 'timeout', .5, os.environ.copy())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_signal_stops_step_before_final_capture(tmp_path):
    marker, final = tmp_path / 'writer', tmp_path / 'final'
    child = (f'from pathlib import Path; import time; p=Path({str(marker)!r}); '
             '\nwhile True: p.write_text(str(time.monotonic())); time.sleep(.01)')
    script = '''import signal,sys
from pathlib import Path
from integrations.autoresearch_classic_job import run_phase
def stop(*args): raise RuntimeError('signal')
signal.signal(signal.SIGTERM, stop)
try: run_phase([sys.executable,'-c',sys.argv[1]],sys.argv[2],60,__import__('os').environ.copy())
finally: Path(sys.argv[3]).write_text('capture')
'''
    process = subprocess.Popen([sys.executable, '-c', script, child, str(tmp_path / 'log'), str(final)])
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=15)
        assert final.read_text() == 'capture'
        captured = marker.read_text()
        time.sleep(.1)
        assert marker.read_text() == captured
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_funded_identity_is_explicit_single_h100():
    env = dict(GENERALS_COMPUTE_PROVIDER='autoresearch', GMN_OUTPUT_DIR='/output', GMN_CPU_LIMIT='8')
    with patch.object(autoresearch_gpu, 'gpu_query', return_value='0, GPU-abcd, NVIDIA H100 80GB HBM3'):
        assert autoresearch_gpu.identity(env)['uuid'] == 'GPU-abcd'
        with pytest.raises(RuntimeError, match='explicit'):
            autoresearch_gpu.identity(dict(env, GENERALS_COMPUTE_PROVIDER='slurm'))
    for row in ('0, GPU-abcd, NVIDIA B300', '0, GPU-abcd, H100\n1, GPU-efab, H100'):
        with patch.object(autoresearch_gpu, 'gpu_query', return_value=row), pytest.raises(RuntimeError):
            autoresearch_gpu.identity(env)


def test_busy_funded_gpu_is_not_used():
    with patch.object(autoresearch_gpu, 'identity', return_value=dict(uuid='GPU-abcd')), patch.object(
            autoresearch_gpu, 'gpu_query', return_value='12345'):
        with pytest.raises(RuntimeError, match='leaving them untouched'):
            autoresearch_gpu.verify_idle()

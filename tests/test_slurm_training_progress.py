"""Host progress is numeric, bounded, and never starts another Slurm step."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from integrations.slurm_s3_job import SlurmJob


def read_progress(root, text, receipt=None):
    path = root / 'out/run/console.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    job = SimpleNamespace(root=root, config={'receipt': receipt or {}})
    return SlurmJob.training_progress(job)


def test_real_dashboard_times_and_throughput_ignore_injected_text():
    with TemporaryDirectory() as directory:
        progress = read_progress(Path(directory), '''private_token=DO_NOT_COPY
╭────
│ Epoch              1177 │
│ Uptime       2m 0s 100ms │
╭────
│ Epoch              1178 │
│ Uptime       2m 25s 100ms │
╭────
│ Epoch              1179 │
│ Uptime       2m 50s 100ms │
https://credential.example/DO_NOT_COPY
''', {'environment_count': 8192, 'horizon': 256})
        assert progress == ('TRAIN_PROGRESS epoch=1179 uptime_seconds=170.100'
                            ' environment_sps_last_two_epochs=83886.08')


def test_partial_dashboard_does_not_claim_new_epoch():
    with TemporaryDirectory() as directory:
        progress = read_progress(Path(directory), '╭ Epoch 5 Uptime 1m 5s\n╭ Epoch 6')
        assert progress == 'TRAIN_PROGRESS epoch=5 uptime_seconds=65.000'


def test_missing_log_is_normal_during_startup():
    job = SimpleNamespace(root=Path('/nonexistent/relh-generals-progress'), config={})
    assert SlurmJob.training_progress(job) is None


def test_bounded_tail_excludes_old_progress():
    with TemporaryDirectory() as directory:
        assert read_progress(Path(directory), '╭ Epoch 5 Uptime 1m 5s\n' + 'x' * 131073) is None


def test_unqualified_batch_settings_omit_sps():
    with TemporaryDirectory() as directory:
        progress = read_progress(Path(directory), '╭ Epoch 1 Uptime 1s\n╭ Epoch 3 Uptime 3s',
                                 {'environment_count': 'secret', 'horizon': 256})
        assert progress == 'TRAIN_PROGRESS epoch=3 uptime_seconds=3.000'

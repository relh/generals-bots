import json
import os
import subprocess
import tarfile
from unittest.mock import patch

import pytest

from integrations.recover_retained_results import recover


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    monkeypatch.setenv("SLURM_JOB_NAME", "relh-generals-recovery-test")
    root = tmp_path / "relh-generals-111"
    (root / "out").mkdir(parents=True)
    (root / "out/checkpoint.bin").write_bytes(b"retained checkpoint")
    log = tmp_path / "relh-generals-original-111.log"
    log.write_text("retained diagnostic")
    return dict(root=str(root), batch_log=str(log), expected_uid=os.getuid(),
                original_receipt=dict(job_id="111", Nice="2147483645", Priority="1",
                                      JobState="FAILED", ExitCode="125:0"),
                result_prefix="owned/recovery", output_urls=["private-put"], manifest_url="private-manifest")


def run_recovery(config, *, active="", controller_error=False, upload_error=False):
    original = subprocess.run
    published = []

    def command(argv, **kwargs):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, int(controller_error), active, "")
        return original(argv, **kwargs)

    def upload(url, path, **kwargs):
        assert kwargs["upload"]
        if upload_error:
            raise RuntimeError("Deliberate bounded upload failure")
        published.append((url, path.read_bytes()))

    with patch("subprocess.run", command), patch("integrations.slurm_s3_job.transfer", upload):
        recover(config)
    return published


def test_archive_only_after_inactive_owner_verified(recovery):
    from pathlib import Path

    root = Path(recovery["root"])
    uploads = run_recovery(recovery)
    assert [url for url, _ in uploads] == ["private-put", "private-manifest"]
    result = json.loads(uploads[-1][1])
    assert result["receipt"]["job_id"] == "111" and result["receipt"]["recovery_job_id"] == "222"
    assert result["receipt"]["workload_exit_code"] == 125
    with tarfile.open(root / "results.tar.gz") as archive:
        assert archive.extractfile("out/checkpoint.bin").read() == b"retained checkpoint"
    assert (root / "out/checkpoint.bin").read_bytes() == b"retained checkpoint"
    with pytest.raises(FileExistsError):
        run_recovery(recovery)


@pytest.mark.parametrize("active", ["111\n", "111.3\n", "111.batch\n"])
def test_active_job_or_observer_prevents_archiving(recovery, active):
    with pytest.raises(RuntimeError, match="still active"):
        run_recovery(recovery, active=active)


def test_unavailable_controller_prevents_archiving(recovery):
    with pytest.raises(RuntimeError, match="unavailable"):
        run_recovery(recovery, controller_error=True)


def test_wrong_owner_prevents_archiving(recovery):
    recovery["expected_uid"] += 1
    with pytest.raises(ValueError, match="not owned"):
        run_recovery(recovery)


def test_upload_failure_preserves_original_and_new_archive(recovery):
    from pathlib import Path

    root = Path(recovery["root"])
    with pytest.raises(RuntimeError, match="upload failure"):
        run_recovery(recovery, upload_error=True)
    assert (root / "out/checkpoint.bin").read_bytes() == b"retained checkpoint"
    assert (root / "results.tar.gz").is_file()

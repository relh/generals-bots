"""Collection must prove identity and remain read-only while a job is live."""
import copy
from types import SimpleNamespace

import pytest

from integrations.collect_policy_results import collect, verify_manifest
from integrations.slurm_s3_job import NICE


def manifest():
    expected = dict(source_revision="source", input_sha256="a"*64, image_sha256="b"*64, result_prefix="owned/results")
    record = dict(receipt=dict(expected, job_id="35933", Nice=str(NICE), Priority="1", workload_exit_code=0),
                  parts=[dict(index=0, bytes=5, sha256="c"*64)])
    return record, expected


def test_manifest_checks_actual_job_and_source():
    record, expected = manifest()
    assert verify_manifest(record, expected, "35933") == 5
    altered = copy.deepcopy(record)
    altered["receipt"]["source_revision"] = "another source"
    with pytest.raises(ValueError, match="source or artifact identity"):
        verify_manifest(altered, expected, "35933")
    with pytest.raises(ValueError, match="job or scheduling identity"):
        verify_manifest(record, expected, "99999")


@pytest.mark.parametrize("part", [dict(index=1, bytes=5, sha256="c"*64),
    dict(index=0, bytes=-1, sha256="c"*64), dict(index=0, bytes=5, sha256="invalid")])
def test_bad_part_metadata_rejected(part):
    record, expected = manifest()
    record["parts"] = [part]
    with pytest.raises(ValueError):
        verify_manifest(record, expected, "35933")


def test_live_job_never_downloads_or_creates_output(tmp_path, monkeypatch):
    from integrations import collect_policy_results as collector

    monkeypatch.setattr(collector, "remote", lambda *a, **kw: SimpleNamespace(returncode=0,
        stdout=f"JobId=35933 JobState=RUNNING ExitCode=0:0 Nice={NICE} Priority=1 TimeLimit=02:00:00"))
    monkeypatch.setattr(collector, "sandbox_session", lambda *a: pytest.fail("Live collection must not touch AWS"))
    output = tmp_path / "new-result"
    result = collect({}, {"job_id": "35933"}, output)
    assert result == dict(job_id="35933", state="RUNNING", collected=False)
    assert not output.exists()

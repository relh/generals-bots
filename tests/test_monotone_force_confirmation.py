"""A failed/missing development gate must never emit runnable confirmation."""
import hashlib
import json
import pytest
from integrations.monotone_force_confirmation import prepare, PLAN

@pytest.mark.parametrize('change', [dict(provider_status='running'), dict(selected=False),
                                    dict(technical_success=False), dict(qualification_verified=False), dict(job_id='job-kzmub')])
def test_unqualified_development_creates_no_configuration(tmp_path,change):
    audit=dict(schema=PLAN['required_terminal_audit_schema'],job_id='job-wgtyc',provider_status='succeeded',
               technical_success=True,qualification_verified=True,collection_complete=True,selected=True)
    audit.update(change)
    path=tmp_path/'audit.json';path.write_text(json.dumps(audit))
    target=tmp_path/'not-created'/'config.json'
    with pytest.raises(ValueError):
        prepare(path,hashlib.sha256(path.read_bytes()).hexdigest(),tmp_path/'source',target)
    assert not target.parent.exists()

def test_missing_evidence_creates_no_configuration(tmp_path):
    target=tmp_path/'not-created'/'config.json'
    with pytest.raises(FileNotFoundError):prepare(tmp_path/'missing.json','0'*64,tmp_path/'source',target)
    assert not target.parent.exists()

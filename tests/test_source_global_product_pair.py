import hashlib
import json

import numpy as np
import pytest

from integrations.policy_execution import runtime_environment
from integrations.source_global_product_pair import qualified_probe, verify_control


def write_bundle(path, *, q=0.0, local=1.0):
    path.mkdir()
    np.savez(path / "weights.npz",
             product_local_kernel=np.full((32, 8), local, np.float32),
             product_global_kernel=np.full((32, 8), 2.0, np.float32),
             product_action_kernel=np.full((8, 8), q, np.float32))


def test_zero_q_control_keeps_source_product_projections(tmp_path):
    source, control = tmp_path / "source", tmp_path / "control"
    write_bundle(source)
    write_bundle(control)
    verify_control(source, control)
    np.savez(control / "weights.npz",
             product_local_kernel=np.full((32, 8), 1.0, np.float32),
             product_global_kernel=np.full((32, 8), 2.0, np.float32),
             product_action_kernel=np.ones((8, 8), np.float32))
    with pytest.raises(ValueError, match="bitwise zero"):
        verify_control(source, control)
    write_bundle(control.parent / "changed", local=1.1)
    with pytest.raises(ValueError, match="untrained Product projection"):
        verify_control(source, control.parent / "changed")


def test_product_mask_is_explicit_per_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("METTA_SPATIAL_PRODUCT_HEAD_FROZEN", "1")
    source = tmp_path / "source"
    output = tmp_path / "output"
    assert runtime_environment(source, output, {}, product_head_frozen=False)[
        "METTA_SPATIAL_PRODUCT_HEAD_FROZEN"] == "0"
    assert runtime_environment(source, output, {}, product_head_frozen=True)[
        "METTA_SPATIAL_PRODUCT_HEAD_FROZEN"] == "1"


def test_qualified_probe_requires_exact_terminal_and_balanced_population(tmp_path):
    marker, terminal = tmp_path / "marker.json", tmp_path / "terminal.json"
    intent = {"source_policy_sha256": "source"}
    record = {"intent": intent, "audit": {
        "environment_steps": 4194304, "steady_sps": 34500, "illegal_actions": 0,
        "reward_audit": {"nonfinite_rewards": 0, "native_clipped_rewards": 0,
                         "native_clipped_terminal_rewards": 0},
        "opponent_counts_by_seat": {str(i): {"0": 2, "1": 2} for i in range(13)},
    }}
    marker.write_text(json.dumps(record))
    terminal.write_text(json.dumps({"job_id": "job-ffctd", "status": "succeeded", "restarts_used": 0}))
    sha = hashlib.sha256(marker.read_bytes()).hexdigest()
    plan = {"required_probe_job_id": "job-ffctd", "required_probe_steps": 4194304,
            "training": {"steady_sps_floor": 30000}}
    assert qualified_probe(marker, terminal, sha, plan, intent) == record
    record["audit"]["opponent_counts_by_seat"]["0"]["1"] = 0
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not qualify"):
        qualified_probe(marker, terminal, hashlib.sha256(marker.read_bytes()).hexdigest(), plan, intent)

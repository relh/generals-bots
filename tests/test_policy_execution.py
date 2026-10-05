import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from integrations.policy_execution import execute, runtime_environment
from integrations.spatial_action_sampling import acting_logits
from integrations.spatial_policy_bundle import structured_action_probabilities


def test_sampler_environment_roundtrip_clears_inherited_experiment_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("METTA_SPATIAL_LOG_GAP_SCALE", "8")
    monkeypatch.setenv("METTA_SPATIAL_ROUTE_HALF_WEIGHT", "1")
    monkeypatch.setenv("METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE", "9")
    monkeypatch.setenv("METTA_SPATIAL_EARLY_ROUTE_TURNS", "999")
    sampler = {"mode": "structured_sample", "move_temperature": 0.05, "split_temperature": 0.15}
    env = runtime_environment(tmp_path, tmp_path, sampler)
    assert "METTA_SPATIAL_LOG_GAP_SCALE" not in env
    assert "METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE" not in env
    assert "METTA_SPATIAL_EARLY_ROUTE_TURNS" not in env
    outputs = np.zeros((1, 3530), np.float32)
    outputs[0, 0], outputs[0, 1764] = 0.1, 0.05
    legal = np.zeros(3529, bool)
    legal[[0, 1764]] = True
    logits = acting_logits(
        outputs, float(env["METTA_SPATIAL_POLICY_TEMPERATURE"]), float(env["METTA_SPATIAL_SPLIT_TEMPERATURE"]), np
    )[0, :3529]
    logits = np.where(legal, logits, -np.inf)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    expected = structured_action_probabilities(outputs[0], legal, 0.05, 0.15)
    np.testing.assert_allclose(probabilities, expected, atol=1e-6)


def test_timeout_stops_stubborn_owned_descendant_and_preserves_unrelated_process(tmp_path, monkeypatch):
    package = tmp_path / "integrations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    child = (
        "import signal,sys,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\np=Path(sys.argv[1])\n"
        "while True:\n p.write_text(str(time.monotonic()))\n time.sleep(.02)\n"
    )
    (package / "worker.py").write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}, sys.argv[1]])\n"
        "time.sleep(30)\n"
    )
    monkeypatch.chdir(tmp_path)
    heartbeat = tmp_path / "heartbeat"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(TimeoutError):
            execute(
                "worker",
                [heartbeat],
                source=tmp_path,
                output=tmp_path / "logs",
                sampler={"mode": "structured_sample", "move_temperature": 0.05, "split_temperature": 0.15},
                name="timeout",
                seconds=0.1,
            )
        assert heartbeat.exists()
        recorded = heartbeat.read_text()
        time.sleep(0.15)
        assert heartbeat.read_text() == recorded
        assert unrelated.poll() is None
        assert signal.getsignal(signal.SIGTERM) == previous
        assert (tmp_path / "logs/timeout-process.log").exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)

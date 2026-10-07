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


def test_nested_process_preserves_existing_outer_phase_log(tmp_path, monkeypatch):
    package = tmp_path / "integrations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "worker.py").write_text("print('native phase completed')\n")
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir()
    outer = logs / "build.log"
    outer.write_bytes(b"outer scheduler phase evidence\n")
    execute("worker", [], source=tmp_path, output=logs,
            sampler={"mode": "structured_sample", "move_temperature": .05, "split_temperature": .15},
            name="build", seconds=5)
    assert outer.read_bytes() == b"outer scheduler phase evidence\n"
    assert (logs / "build-process.log").read_text() == "native phase completed\n"


def test_progress_visible_before_child_exit_and_error_tail_drained_once(tmp_path, monkeypatch):
    import io
    import integrations.policy_execution as execution

    package = tmp_path / "integrations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    acknowledgement = tmp_path / "progress-seen"
    (package / "worker.py").write_text(
        "import sys,time\nfrom pathlib import Path\n"
        "print('x' * 65535 + 'é\\nPROGRESS turn=50')\n"
        "deadline=time.monotonic()+6\n"
        "while not Path(sys.argv[1]).exists():\n"
        " if time.monotonic()>deadline: raise RuntimeError('progress was not live')\n"
        " time.sleep(.02)\n"
        "print('ERROR final tail ' + 'z' * 70000, file=sys.stderr)\n"
        "sys.exit(7)\n"
    )

    class ObservedOutput(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "PROGRESS turn=50" in self.getvalue():
                acknowledgement.touch()
            return result

    visible = ObservedOutput()
    monkeypatch.setattr(execution.sys, "stdout", visible)
    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "logs"
    with pytest.raises(RuntimeError, match="progress exited 7"):
        execute("worker", [acknowledgement], source=tmp_path, output=logs,
                sampler={"mode": "structured_sample", "move_temperature": .05, "split_temperature": .15},
                name="progress", seconds=8)
    assert acknowledgement.exists()
    retained = (logs / "progress-process.log").read_bytes()
    assert visible.getvalue() == retained.decode("utf-8")
    assert visible.getvalue().count("PROGRESS turn=50") == 1
    assert visible.getvalue().endswith("ERROR final tail " + "z" * 70000 + "\n")


def test_native_training_console_is_live_and_drained_when_monitor_fails(tmp_path, monkeypatch):
    import io
    import json
    import integrations.policy_execution as execution

    package = tmp_path / "integrations"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "worker.py").write_text(
        "import signal,sys,time\nfrom pathlib import Path\n"
        "path=Path(sys.argv[1]); path.parent.mkdir()\n"
        "log=path.open('w')\n"
        "def stop(signum,frame):\n print('native shutdown tail',file=log,flush=True); sys.exit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        "print('NATIVE epoch progress',file=log,flush=True)\n"
        "deadline=time.monotonic()+4\n"
        "while not Path(sys.argv[2]).exists():\n"
        " if time.monotonic()>deadline: raise RuntimeError('native progress was not live')\n"
        " time.sleep(.02)\n"
        "print('NonFiniteGradsError: native failure',file=log,flush=True)\n"
        "time.sleep(30)\n"
    )
    acknowledgement = tmp_path / "native-progress-seen"

    class ObservedOutput(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if "NATIVE epoch progress" in self.getvalue():
                acknowledgement.touch()
            return result

    visible = ObservedOutput()
    monkeypatch.setattr(execution.sys, "stdout", visible)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "training.json"
    config.write_text(json.dumps({"total_timesteps": 32, "overrides": {
        "vec.total_agents": 8, "train.horizon": 4}}))
    logs = tmp_path / "logs"
    console = logs / "run/console.log"
    with pytest.raises(FloatingPointError, match="Native training produced nonfinite"):
        execute("worker", [console, acknowledgement], source=tmp_path, output=logs,
                sampler={"mode": "structured_sample", "move_temperature": .05, "split_temperature": .15},
                name="train", seconds=8, training_config=config)
    assert acknowledgement.exists()
    assert visible.getvalue() == console.read_text()
    assert visible.getvalue().count("NATIVE epoch progress") == 1
    assert visible.getvalue().endswith("native shutdown tail\n")
    assert (logs / "train-process.log").read_bytes() == b""


@pytest.mark.parametrize("scale", ["0", "4"])
def test_retired_sampler_environment_is_rejected(tmp_path, monkeypatch, scale):
    from integrations.launch_spatial_selfplay_training import rollout_sampler_settings

    monkeypatch.setenv("METTA_SPATIAL_LOG_GAP_SCALE", scale)
    sampler = dict(mode="structured_sample", move_temperature=.05, split_temperature=.15)
    with pytest.raises(ValueError, match="retired sampler"):
        runtime_environment(tmp_path, tmp_path, sampler)
    with pytest.raises(ValueError, match="retired sampler"):
        rollout_sampler_settings({"METTA_SPATIAL_LOG_GAP_SCALE": scale})

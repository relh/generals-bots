"""The facade must preserve the native bootstrap and optimizer continuation."""
import json

import pytest

from integrations.policy import command_spec


def test_preflight_uses_fresh_cpu_bootstrap_and_keeps_arguments():
    arguments = ["--build", "/build", "--config", "/run.json", "--output", "/new"]
    argv, env = command_spec("preflight", arguments, {"PYTHONPATH": "/framework"})
    assert argv[2:4] == ["integrations.launch_spatial_selfplay_training", "preflight"]
    assert argv[4:] == arguments
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["METTA_MEMORYLESS_OPTIMIZATION"] == "1"
    assert env["PYTHONPATH"].split(":")[0].endswith("puffer_bootstrap")
    assert env["PYTHONPATH"].endswith("/framework")


def test_resume_rejects_policy_only_transfer(tmp_path):
    config = tmp_path / "run.json"
    config.write_text(json.dumps({"initialize": {"restore_learner": False}}))
    with pytest.raises(ValueError, match="restore_learner"):
        command_spec("resume", ["--config", str(config)], {})
    config.write_text(json.dumps({"initialize": {"restore_learner": True}}))
    argv, _ = command_spec("resume", ["--config", str(config)], {})
    assert argv[3] == "train"


def test_evaluation_preserves_sampler_options_and_environment():
    arguments = ["--bundle", "/actor", "--seed", "72", "--sample-seed", "81"]
    env = {"JAX_PLATFORMS": "cuda,cpu"}
    argv, actual = command_spec("evaluate", arguments, env)
    assert argv[2:] == ["integrations.evaluate_spatial_population", *arguments]
    assert actual == env

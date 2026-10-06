import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from integrations.matched_defense_experiment import prepare_configs, validate_qualification, verify_runtime


def configs():
    return {
        "fabric": {"observation_size": 7056, "action_sizes": [3529]},
        "python_environment": {
            "options": {"parallel_games": 8192, "horizon": 2000, "shaping_gamma": .999,
                        "balance_opponent_sides": True, "coworld_position_probability": .25},
            "spec": {"agents": 8192, "observation_size": 7056, "action_sizes": [3529]},
        },
    }, {
        "seed": 8857, "total_timesteps": 8388608,
        "initialize": {"asset": "/work/input/assets/cold/asset.json", "manifest_sha256": "a" * 64,
                       "restore_learner": False},
        "overrides": {"vec.total_agents": 8192, "train.horizon": 256,
                      "train.minibatch_size": 8192, "train.replay_ratio": .5, "train.gamma": .999},
    }


def test_geometry_projection_preserves_objective_and_equal_update_budget():
    build, run = configs()
    original = copy.deepcopy((build, run))
    for agents, horizon in ((2048, 256), (4096, 128)):
        target, learner = prepare_configs(build, run, agents=agents, horizon=horizon,
                                          steps=8388608, seed=8857)
        assert target["python_environment"]["options"]["horizon"] == 2000
        assert target["python_environment"]["options"]["shaping_gamma"] == learner["overrides"]["train.gamma"]
        assert learner["total_timesteps"] * learner["overrides"]["train.replay_ratio"] / 8192 == 512
    assert (build, run) == original
    with pytest.raises(ValueError, match="complete rollouts"):
        prepare_configs(build, run, agents=2048, horizon=256, steps=8388609, seed=8857)


def test_qualification_rejects_wrong_batching_and_native_runtime(tmp_path):
    build, run = prepare_configs(*configs(), agents=2048, horizon=256, steps=8388608, seed=8857)
    directory = tmp_path / "probe"
    (directory / "pilot").mkdir(parents=True)
    audit = dict(environment_count=2048, horizon=256, minibatch=8192, replay_ratio=.5,
                 illegal_actions=0, steady_sps=40000, environment_steps=3145728)
    result = dict(qualified=True, audit=audit)
    for path, value in (("result.json", result), ("pilot/training-audit.json", audit),
                        ("build-config.json", build), ("pilot/config.json", run)):
        (directory / path).write_text(json.dumps(value))
    digest = hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    validate_qualification(directory, digest, build, run)
    changed = copy.deepcopy(run)
    changed["overrides"]["train.minibatch_size"] = 4096
    with pytest.raises(ValueError, match="batching"):
        validate_qualification(directory, digest, build, changed)
    for name in ("before", "after"):
        root = tmp_path / name
        (root / "source/src").mkdir(parents=True)
        (root / "source/src/kernel.cu").write_text("same kernel")
        (root / "build.json").write_text(json.dumps(dict(revision="r", model_sha256="m",
                                                       environment_sha256="e", config=build)))
    verify_runtime(tmp_path / "before", tmp_path / "after")
    (tmp_path / "after/source/src/kernel.cu").write_text("different mask gather")
    with pytest.raises(ValueError, match="runtime source"):
        verify_runtime(tmp_path / "before", tmp_path / "after")


def test_trial_runtime_gpu_guard_needs_no_slurm_assignment(monkeypatch, tmp_path):
    from integrations import slurm_s3_job
    from integrations.policy_trial import Trial

    for key in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS", "GENERALS_ALLOCATED_GPU_UUID"):
        monkeypatch.delenv(key, raising=False)
    def query(*arguments):
        if "--query-gpu=index,uuid" in arguments:
            return "0, GPU-aabb-1234"
        if "--query-compute-apps=pid" in arguments:
            return ""
        return "GPU-aabb-1234, 0, 0"
    monkeypatch.setattr(slurm_s3_job, "gpu_query", query)
    # A provider without Slurm must reach and enforce the JAX device-count guard.
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda platform: [object(), object()]))
    trial = Trial.__new__(Trial)
    trial.output = tmp_path
    with pytest.raises(RuntimeError, match="exactly one allocated GPU"):
        trial.smoke()


def test_training_startup_fits_cold_h100_compilation_and_finite_outer_budget(monkeypatch, tmp_path):
    from integrations import policy_trial

    requests = []
    monkeypatch.setattr(policy_trial, "execute", lambda *args, **kwargs: requests.append(kwargs))
    trial = policy_trial.Trial.__new__(policy_trial.Trial)
    trial.output, trial.source, trial.sampler = tmp_path, tmp_path / "source", {}
    trial.call("launch_spatial_selfplay_training", [], name="train", seconds=660,
               arm="control", training=True)
    assert requests[0]["startup_seconds"] == 420 < requests[0]["seconds"] == 660
    assert requests[0]["training_config"] == tmp_path / "control/config.json"
    trial.call("evaluate_spatial_frozen_match", [], name="sampling", seconds=300)
    assert requests[1]["training_config"] is None
    assert requests[1]["seconds"] == 300

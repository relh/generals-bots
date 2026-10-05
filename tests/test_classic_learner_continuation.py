import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from integrations.classic_learner_continuation import load_continuation, write_manifest
from integrations.learner_checkpoint import LEARNER_HEADER


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def current(tmp_path):
    root = tmp_path / "inputs"
    parent = root / "parent"
    run = parent / "run"
    checkpoint = run / "checkpoints/metta_generals/run/best.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(np.array([0.1, 0.2], np.float32).tobytes())
    options = {
        "parallel_games": 32,
        "horizon": 2000,
        "shaping_gamma": 0.999,
        "balance_opponent_sides": True,
        "coworld_classic": True,
        "teacher": None,
        "frozen_bundle": "original-zero",
        "frozen_bundles": ["original-zero", "original-one"],
        "scripted_opponents": ["expander_harvester", "sentinel"],
        "opponent_weights": [1, 1, 1, 1],
    }
    record = {
        "build": {
            "config": {
                "python_environment": {
                    "options": options,
                    "spec": {"agents": 32, "observation_size": 7056, "action_sizes": [3529]},
                },
                "fabric": {"observation_size": 7056, "action_sizes": [3529]},
            }
        },
        "config": {
            "seed": 54321,
            "total_timesteps": 1024,
            "overrides": {
                "vec.total_agents": 32,
                "train.horizon": 8,
                "train.minibatch_size": 32,
                "train.learning_rate": 0.0002,
                "train.gamma": 0.999,
            },
        },
    }
    (run / "training.json").write_text(json.dumps(record))
    learner = Path(str(checkpoint) + ".learner")
    learner.write_bytes(LEARNER_HEADER.pack(b"METTAL01", 4, 1024, 2, 0.0002) + np.zeros(2, np.float32).tobytes())
    identity = {
        "policy_sha256": digest(checkpoint),
        "state_sha256": digest(learner),
        "run_sha256": digest(run / "training.json"),
        "environment_sha256": [],
    }
    Path(str(checkpoint) + ".learner.json").write_text(json.dumps(identity))

    def bundle(path, policy):
        path.mkdir(parents=True)
        (path / "policy.bin").write_bytes(policy)
        (path / "training.json").write_bytes((run / "training.json").read_bytes())
        (path / "build.json").write_text("{}")
        (path / "weights.npz").write_bytes(b"unit tensor fixture")
        files = {name: digest(path / name) for name in ("policy.bin", "training.json", "build.json", "weights.npz")}
        (path / "spatial-policy.json").write_text(
            json.dumps({"schema": "puffer5-generals-spatial-v1", "files": files, "factory_source_sha256": "0" * 64})
        )

    bundle(parent / "bundle", checkpoint.read_bytes())
    frozen = [root / "frozen/0", root / "frozen/1"]
    for index, path in enumerate(frozen):
        bundle(path, np.array([index + 1], np.float32).tobytes())
    manifest = tmp_path / "current.json"
    write_manifest(root, frozen, manifest)
    return root, manifest, checkpoint


def test_authentic_clock_seed_pool_and_source_files_preserved(current):
    root, manifest, _ = current
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    _, build, run, audit = load_continuation(manifest, 768)
    assert run["seed"] == 54321 and run["total_timesteps"] == 1792
    assert run["initialize"]["restore_learner"] is True
    assert audit["starting_agent_steps"] == 1024
    assert build["python_environment"]["options"]["opponent_weights"] == [1, 1, 1, 1]
    assert "coworld_classic" not in build["python_environment"]["options"]
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("budget", [0, -256, 513, True])
def test_budget_uses_actual_rollout_not_historical_whitelist(current, budget):
    with pytest.raises(ValueError, match="aligned"):
        load_continuation(current[1], budget)


def test_unknown_recipe_and_escaping_binding_rejected(current):
    _, path, _ = current
    original = json.loads(path.read_text())
    for mutate in (
        lambda value: value.update(context_extension="preserve"),
        lambda value: value["frozen_bundles"][0].update(path="../outside"),
    ):
        value = copy.deepcopy(original)
        mutate(value)
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            load_continuation(path, 256)


def test_corrupted_optimizer_or_pool_rejected(current):
    _, path, checkpoint = current
    learner = Path(str(checkpoint) + ".learner")
    learner.write_bytes(learner.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="identity"):
        load_continuation(path, 256)


def test_authenticated_but_wrong_optimizer_clock_rejected(current):
    _, path, checkpoint = current
    learner = Path(str(checkpoint) + ".learner")
    raw = bytearray(learner.read_bytes())
    struct.pack_into("<Q", raw, 8, 3)
    learner.write_bytes(raw)
    manifest = json.loads(path.read_text())
    manifest["learner_sha256"] = digest(learner)
    path.write_text(json.dumps(manifest))
    identity_path = Path(str(checkpoint) + ".learner.json")
    identity = json.loads(identity_path.read_text())
    identity["state_sha256"] = digest(learner)
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="clock"):
        load_continuation(path, 256)

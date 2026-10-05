"""Authenticate one current Classic policy/optimizer and preserve its continuation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path

from integrations.classic_contract import project_current_options, validate_training_contract
from integrations.learner_checkpoint import LearnerCheckpoint

SCHEMA = "generals-current-continuation-v1"
FIELDS = {
    "schema",
    "root",
    "parent",
    "checkpoint",
    "policy_sha256",
    "learner_sha256",
    "training_sha256",
    "bundle_manifest_sha256",
    "frozen_bundles",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads((path / "spatial-policy.json").read_text())
    if manifest["schema"] != "puffer5-generals-spatial-v1" or set(manifest["files"]) != {
        "build.json",
        "training.json",
        "policy.bin",
        "weights.npz",
    }:
        raise ValueError("Current spatial bundle fields differ")
    for name, digest in manifest["files"].items():
        if sha(path / name) != digest:
            raise ValueError("Bundle artifact checksum differs: " + name)
    return manifest


def verify_factory_source(parent, source):
    expected = verify_bundle(Path(parent) / "bundle")["factory_source_sha256"]
    if sha(source) != expected:
        raise ValueError("Continuation factory source differs from the frozen checkpoint")
    return expected


def relative_path(root, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Continuation references must be relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Continuation reference escapes its authenticated input root")
    return path


def write_manifest(root, frozen_bundles, output):
    """Author current bindings from actual artifacts; source records are never rewritten."""
    root, output = Path(root).resolve(), Path(output)
    parent = root / "parent"
    bundle = verify_bundle(parent / "bundle")
    matching = [
        path for path in (parent / "run/checkpoints").rglob("*.bin") if sha(path) == bundle["files"]["policy.bin"]
    ]
    if len(matching) != 1:
        raise ValueError("Parent bundle must identify one unambiguous authentic checkpoint")
    checkpoint = matching[0]
    record = json.loads((parent / "run/training.json").read_text())
    if record != json.loads((parent / "bundle/training.json").read_text()):
        raise ValueError("Parent run and serving bundle training records differ")
    if len(frozen_bundles) != len(record["build"]["config"]["python_environment"]["options"]["frozen_bundles"]):
        raise ValueError("Current frozen bindings must preserve the authenticated pool size")
    entries = []
    for path in frozen_bundles:
        path = Path(path).resolve()
        manifest = verify_bundle(path)
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "policy_sha256": manifest["files"]["policy.bin"],
                "manifest_sha256": sha(path / "spatial-policy.json"),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "root": str(root),
        "parent": "parent",
        "checkpoint": str(checkpoint.relative_to(root)),
        "policy_sha256": sha(checkpoint),
        "learner_sha256": sha(str(checkpoint) + ".learner"),
        "training_sha256": sha(parent / "run/training.json"),
        "bundle_manifest_sha256": sha(parent / "bundle/spatial-policy.json"),
        "frozen_bundles": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as file:
        file.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_continuation(manifest_path, additional_steps):
    manifest = json.loads(Path(manifest_path).read_text())
    if set(manifest) != FIELDS or manifest["schema"] != SCHEMA:
        raise ValueError("Require the current continuation manifest without recipes or unexpected fields")
    root = Path(manifest["root"]).resolve()
    parent, checkpoint = relative_path(root, manifest["parent"]), relative_path(root, manifest["checkpoint"])
    if not checkpoint.is_relative_to(parent / "run/checkpoints"):
        raise ValueError("Continuation checkpoint must belong to its actual parent run")
    for key, path in (
        ("policy_sha256", checkpoint),
        ("learner_sha256", str(checkpoint) + ".learner"),
        ("training_sha256", parent / "run/training.json"),
        ("bundle_manifest_sha256", parent / "bundle/spatial-policy.json"),
    ):
        if sha(path) != manifest[key]:
            raise ValueError("Continuation identity differs: " + key)
    bundle = verify_bundle(parent / "bundle")
    if bundle["files"]["policy.bin"] != manifest["policy_sha256"]:
        raise ValueError("Exported parent policy differs from the source checkpoint")
    if sha(parent / "bundle/training.json") != manifest["training_sha256"]:
        raise ValueError("Parent bundle and actual training record differ")
    parameters = checkpoint.read_bytes()
    if (
        not parameters
        or len(parameters) % 4
        or not all(math.isfinite(v) for (v,) in struct.iter_unpack("<f", parameters))
    ):
        raise ValueError("Policy checkpoint must be a finite float32 vector")
    identity = json.loads(Path(str(checkpoint) + ".learner.json").read_text())
    for key, value in (
        ("policy_sha256", manifest["policy_sha256"]),
        ("state_sha256", manifest["learner_sha256"]),
        ("run_sha256", manifest["training_sha256"]),
        ("environment_sha256", []),
    ):
        if identity[key] != value:
            raise ValueError("Device-resident optimizer identity differs: " + key)
    learner = LearnerCheckpoint.read(Path(str(checkpoint) + ".learner"), len(parameters) // 4)
    record = json.loads((parent / "run/training.json").read_text())
    build, run = copy.deepcopy(record["build"]["config"]), copy.deepcopy(record["config"])
    validate_training_contract(build, run)
    geometry = run["overrides"]
    batch = geometry["vec.total_agents"] * geometry["train.horizon"]
    if learner.agent_steps <= 0 or learner.epoch * batch != learner.agent_steps:
        raise ValueError("Actual optimizer clock differs from the source rollout geometry")
    rate = struct.unpack("<f", struct.pack("<f", geometry["train.learning_rate"]))[0]
    if learner.learning_rate != rate:
        raise ValueError("Actual optimizer learning rate differs from the source run")
    if (
        isinstance(additional_steps, bool)
        or not isinstance(additional_steps, int)
        or additional_steps <= 0
        or additional_steps % batch
    ):
        raise ValueError("Continuation budget must be positive and aligned to the actual rollout")
    options = project_current_options(build["python_environment"]["options"])
    entries = manifest["frozen_bundles"]
    if len(entries) != len(options["frozen_bundles"]):
        raise ValueError("Frozen pool size differs from the source configuration")
    paths, digests = [], []
    for entry in entries:
        if set(entry) != {"path", "policy_sha256", "manifest_sha256"}:
            raise ValueError("Current frozen binding fields differ")
        path = relative_path(root, entry["path"])
        if (
            sha(path / "spatial-policy.json") != entry["manifest_sha256"]
            or verify_bundle(path)["files"]["policy.bin"] != entry["policy_sha256"]
        ):
            raise ValueError("Frozen opponent identity differs")
        paths.append(str(path))
        digests.append(entry["policy_sha256"])
    if len(set(digests)) != len(digests):
        raise ValueError("Frozen opponents must have distinct policy checkpoints")
    if len(options["opponent_weights"]) != len(paths) + len(options["scripted_opponents"]):
        raise ValueError("Opponent weights differ from the authenticated pool")
    options.update(frozen_bundles=paths, frozen_bundle=paths[0])
    build["python_environment"]["options"] = options
    run["total_timesteps"] = learner.agent_steps + additional_steps
    run["initialize"] = dict(
        run=str(parent / "run"),
        checkpoint=str(checkpoint),
        sha256=manifest["policy_sha256"],
        restore_learner=True,
        allow_environment_transfer=True,
    )
    return (
        parent,
        build,
        run,
        {
            "starting_agent_steps": learner.agent_steps,
            "additional_steps": additional_steps,
            "ending_agent_steps": learner.agent_steps + additional_steps,
            "policy_sha256": manifest["policy_sha256"],
            "learner_sha256": manifest["learner_sha256"],
            "restore_learner": True,
            "frozen_policy_sha256": digests,
            "environment_resume": "Fresh episodes; device state is not checkpointed",
        },
    )

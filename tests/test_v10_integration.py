"""V10 aliases preserve parent selection and unconditional import provenance."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from competition.agents.sentinel_python.main import make_agent
from generals.evaluation import replay
from generals.evaluation.arena import Rules
from generals.evaluation.cli import V10_OPTIONS, agent
from scripts.build_sentinel_bundle import build
from scripts.compare_agent_runs import compare
from scripts.strategy_arena import candidate
from tests.test_compare_agent_runs import write_run

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("sentinel-v10", 9, True),
    ("sentinel-v10-disabled", 9, False),
    ("sentinel-v10-v6", 6, True),
    ("sentinel-v10-v6-disabled", 6, False),
]
MODULES = [
    "sentinel",
    "sentinel_v3",
    "sentinel_v5",
    "sentinel_v6",
    "sentinel_v7",
    "sentinel_v8",
    "sentinel_v9",
    "sentinel_v10",
]


@pytest.mark.parametrize("alias,parent,enabled", CASES)
def test_v10_all_factories_and_snapshot_options(monkeypatch, alias, parent, enabled):
    rules = Rules(build_castles=True, deathtouch_turn=800, max_turns=1200)
    source = ROOT / "generals/agents/sentinel_v10_agent.py"
    live, decision = replay.make_policy(alias, rules)
    snapshot, snapshot_decision = replay.make_policy(alias, rules, source=source)
    monkeypatch.setenv("SENTINEL_VARIANT", alias.removeprefix("sentinel-"))
    monkeypatch.setenv("SENTINEL_MODE", "competition")
    assert decision is None and snapshot_decision is None
    for policy in (live, snapshot, candidate(alias, rules), make_agent()):
        assert policy.parent_version == parent and policy.mobilize_home is enabled
        assert policy.deathtouch_turn == 800
        assert policy.base.build_castles is True and policy.base.max_turns == 1200
        leaves = jax.tree.leaves(policy.initial_memory((18, 21)))
        assert len(leaves) == (7 if parent == 6 else 19)
        assert all(x.shape == () and x.dtype == jnp.int32 for x in leaves)
    assert V10_OPTIONS[alias] == dict(parent_version=parent, mobilize_home=enabled)
    assert agent(alias, rules, options={"mobilize_home": not enabled}).mobilize_home is not enabled
    with pytest.raises(ValueError, match="Unsupported policy options"):
        agent(alias, rules, options={"commit_defense": False})
    with pytest.raises(ValueError, match="parent_version"):
        agent(alias, rules, options={"parent_version": 8})


@pytest.mark.parametrize("alias,parent,enabled", CASES)
def test_v10_bundle_full_closure_and_isolated_constructor(tmp_path, alias, parent, enabled):
    variant = alias.removeprefix("sentinel-")
    path = tmp_path / "bundle.zip"
    result = build(path, variant=variant)
    assert (
        result["policy_sha256"]
        == hashlib.sha256((ROOT / "generals/agents/sentinel_v10_agent.py").read_bytes()).hexdigest()
    )
    directory = tmp_path / "extracted"
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["variant"] == variant
        for module in MODULES:
            name = f"generals/agents/{module}_agent.py"
            assert archive.read(name) == (ROOT / name).read_bytes()
            assert manifest["source_sha256"][name] == hashlib.sha256(archive.read(name)).hexdigest()
        for script in ("run.sh", "build.sh"):
            assert f"export SENTINEL_VARIANT={variant}\n".encode() in archive.read(script)
        archive.extractall(directory)
    # Import and construct from the archive with no editable repository path.
    # No policy compilation or action inference is performed by this test.
    code = (
        "from main import make_agent; import jax; "
        "p=make_agent(); "
        f"assert p.parent_version=={parent}; assert p.mobilize_home is {enabled}; "
        f"assert len(jax.tree.leaves(p.initial_memory((18,21))))=={7 if parent == 6 else 19}"
    )
    env = os.environ | {
        "PYTHONPATH": "",
        "PYTHONNOUSERSITE": "1",
        "JAX_PLATFORMS": "cpu",
        "SENTINEL_VARIANT": variant,
        "SENTINEL_MODE": "competition",
    }
    env.pop("LD_LIBRARY_PATH", None)
    subprocess.run([sys.executable, "-c", code], cwd=directory, env=env, check=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("alias,parent,enabled", CASES)
def test_v10_snapshot_guard_keeps_every_imported_parent(tmp_path, monkeypatch, alias, parent, enabled):
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    names = [f"generals/agents/{module}_agent.py" for module in MODULES]
    recorded = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original " + name)
        recorded[name] = replay.file_hash(path)
    snapshot = tmp_path / "snapshot.py"
    snapshot.write_bytes((tmp_path / names[-1]).read_bytes())
    for name in names:
        (tmp_path / name).write_text("changed " + name)
    proof = replay.source_provenance(dict(source_hashes=recorded), dict(candidate=alias, opponent="hunter"), snapshot)
    assert set(proof["critical_changed_sources"]) == set(names[:-1])


@pytest.mark.parametrize("alias,parent,enabled", CASES)
@pytest.mark.parametrize("dependency", ["sentinel_v5", "sentinel_v9"])
def test_v10_opponent_comparison_requires_even_disabled_imports(tmp_path, alias, parent, enabled, dependency):
    left, right = tmp_path / "left", tmp_path / "right"
    for directory in (left, right):
        write_run(directory, "win", local=True)
        path = directory / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["opponents"] = [alias]
        metadata["opponent_options"] = V10_OPTIONS[alias]
        metadata["source_hashes"].update({f"generals/agents/{module}_agent.py": "fixture-sha" for module in MODULES})
        path.write_text(json.dumps(metadata))
        csv = directory / "games.csv"
        csv.write_text(csv.read_text().replace("hunter", alias))
    assert compare(left, right, resamples=10)["matchups"]
    path = right / "metadata.json"
    metadata = json.loads(path.read_text())
    del metadata["source_hashes"][f"generals/agents/{dependency}_agent.py"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing"):
        compare(left, right, resamples=10)

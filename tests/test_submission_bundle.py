"""Standalone packaging and stdio deadline accounting contracts."""

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_sentinel_bundle import build

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_is_deterministic_and_contains_frozen_policy(tmp_path):
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    build(first)
    build(second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as bundle:
        policy = bundle.read("generals/agents/sentinel_agent.py")
        manifest = json.loads(bundle.read("manifest.json"))
        assert policy == (ROOT / "generals/agents/sentinel_agent.py").read_bytes()
        assert manifest["source_sha256"]["generals/agents/sentinel_agent.py"] == hashlib.sha256(policy).hexdigest()
        assert "env" not in bundle.read("generals/__init__.py").decode()
        assert bundle.getinfo("run.sh").external_attr >> 16 & 0o111


def test_probe_faults_late_reply_and_does_not_reuse_it(tmp_path):
    fake = """import sys,time
_,h,w=map(int,sys.stdin.readline().split())
for i in range(3):
    for _ in range(1+3*h):
        if not sys.stdin.readline(): sys.exit(0)
    if i==1: time.sleep(.25)
    print('bad' if i==2 else '1 0 0 0 0',flush=True)
"""
    bundle_path, output = tmp_path / "fake.zip", tmp_path / "report.json"
    with zipfile.ZipFile(bundle_path, "w") as bundle:
        bundle.writestr("run.sh", '#!/bin/sh\ncd "$(dirname "$0")"\nexec python3 -u main.py\n')
        bundle.writestr("main.py", fake)
        bundle.writestr("manifest.json", "{}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/probe_sentinel_bundle.py"),
            str(bundle_path),
            "--python",
            sys.executable,
            "--shape",
            "4x4",
            "--frames",
            "3",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    report = json.loads(output.read_text())
    assert report["faults"] == 2
    assert report["frames"][1]["late_reply_drained"]
    assert report["frames"][2]["malformed_reply"] == "bad"
    assert not report["all_deadlines_met"]


def test_cache_build_is_complete_and_can_be_disabled(tmp_path):
    cached, cold = tmp_path / "cached.zip", tmp_path / "cold.zip"
    build(cached)
    build(cold, prewarm_cache=False)
    with zipfile.ZipFile(cached) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["prewarm_shapes"] == [[h, w] for h in range(18, 22) for w in range(18, 22)]
        assert "warm_cache.py" in archive.namelist()
        assert b"JAX_ENABLE_COMPILATION_CACHE=true" in archive.read("build.sh")
        assert b"JAX_ENABLE_COMPILATION_CACHE=true" not in archive.read("run.sh")
    with zipfile.ZipFile(cold) as archive:
        assert "build.sh" not in archive.namelist()
        assert not json.loads(archive.read("manifest.json"))["prewarm_cache"]


def test_v3_bundle_pins_variant_for_build_and_runtime(tmp_path):
    path = tmp_path / "memory.zip"
    build(path, variant="v3-memory")
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("manifest.json"))["variant"] == "v3-memory"
        assert (
            archive.read("generals/agents/sentinel_v3_agent.py")
            == (ROOT / "generals/agents/sentinel_v3_agent.py").read_bytes()
        )
        for script in ("build.sh", "run.sh"):
            assert b"export SENTINEL_VARIANT=v3-memory\n" in archive.read(script)
            assert b"export SENTINEL_MODE=competition\n" in archive.read(script)


@pytest.mark.parametrize("version", [4, 5, 6])
def test_bundle_retains_scoring_dependencies_and_variant(tmp_path, version):
    variant = f"v{version}"
    path = tmp_path / f"{variant}.zip"
    report = build(path, variant=variant)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        modules = ["sentinel", "sentinel_v3", f"sentinel_{variant}"]
        if version == 6:
            modules.append("sentinel_v5")
        for module in modules:
            name = f"generals/agents/{module}_agent.py"
            assert archive.read(name) == (ROOT / name).read_bytes()
        assert report["policy_sha256"] == manifest["source_sha256"][f"generals/agents/sentinel_{variant}_agent.py"]
        for script in ("build.sh", "run.sh"):
            assert f"export SENTINEL_VARIANT={variant}\n".encode() in archive.read(script)

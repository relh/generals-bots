"""Build-time CPU gate for the frozen 2048-game H100 qualification kit."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path


def main():
    signal.alarm(240)
    kit = Path("/opt/pilot")
    work = Path("/work")
    work.mkdir(exist_ok=True)
    assert not any(work.iterdir()), "CPU proof scratch must be empty"
    (work / "out").mkdir()
    (work / "tmp").mkdir()
    (work / "input").mkdir()
    expected = json.loads((kit / "expected-dependencies.json").read_text())
    actual = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    assert actual == expected, "Image dependency versions differ from the qualified training image"
    from integrations.slurm_s3_job import check_space, extract_input

    package = json.loads((kit / "input-package.json").read_text())
    assert hashlib.sha256((kit / "input.tar.gz").read_bytes()).hexdigest() == package["archive_sha256"]
    check_space(work, 2 * package["input_unpacked_bytes"], 2 * package["input_members"] + 1000)
    extract_input(kit / "input.tar.gz", work / "input", package["input_unpacked_bytes"], package["input_members"])
    for name, digest in json.loads((work / "input/source-manifest.json").read_text()).items():
        assert hashlib.sha256((work / "input" / name).read_bytes()).hexdigest() == digest, name
    Path("/recovery").symlink_to(work / "input/recovery", target_is_directory=True)
    sys.path.insert(0, str(work / "input/source"))
    os.environ.update(
        GENERALS_PILOT_PARALLEL_GAMES="2048",
        GENERALS_PILOT_STEPS="8388608",
        GENERALS_FULL_ACTION_TEMPERATURE="1",
        GENERALS_LOG_GAP_SCALE="4",
        GENERALS_EVAL_SEED="40913",
        GENERALS_EVAL_SAMPLE_SEED="10211",
        GENERALS_PILOT_CONTINUATION_MANIFEST="/work/input/continuation/manifest.json",
        GENERALS_PILOT_POSITION_MANIFEST="/work/input/curriculum/manifest.json",
    )
    from integrations import portable_classic_pilot as pilot
    from integrations.puffer_coworld_frozen_transfer import BuildConfig, BuildManifest, RunConfig, prepare_run

    build, config = pilot.prepare_configs()
    fixture = work / "out/build"
    shutil.copytree(kit / "build", fixture)
    built = BuildManifest.model_validate_json((fixture / "build.json").read_text())
    assert built.config == BuildConfig.model_validate(build)
    # Match the exact runtime wrapper's stricter JSON binding as well.
    assert json.loads((fixture / "build.json").read_text())["config"] == build
    assert hashlib.sha256((fixture / "puffer").read_bytes()).hexdigest() == built.binary_sha256
    prepared = prepare_run(fixture, work / "out/prepare", RunConfig.model_validate(config))
    source = work / "input/continuation/parent/run/checkpoints/metta_generals/run/0000002499805184.bin"
    original = Path(str(source) + ".learner").read_bytes()
    assert prepared.initial_parameters == source.read_bytes()
    assert prepared.initial_learner[:8] == original[:8] and prepared.initial_learner[16:] == original[16:]
    assert prepared.batch_steps == 524288
    assert prepared.initialization.rollout_migration["target_epoch"] == 4768
    assert (
        hashlib.sha256(prepared.initial_learner).hexdigest()
        == "b7ea14edd850275f3e156390621e1b70c81a5e7691c0f0c0ba1653140ee92b97"
    )
    from integrations.autoresearch_classic_job import run_phase

    probe = work / "lifecycle"
    probe.mkdir()
    run_phase([sys.executable, "-c", 'print("success")'], probe / "success.log", 10, os.environ.copy())
    try:
        run_phase(
            [sys.executable, "-c", 'print("failure evidence"); exit(6)'], probe / "failure.log", 10, os.environ.copy()
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Failed phase accepted")
    assert "failure evidence" in (probe / "failure.log").read_text()
    pid = probe / "pid"
    child = (
        "import os,signal,time; from pathlib import Path; Path("
        + repr(str(pid))
        + ").write_text(str(os.getpid())); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    )
    try:
        run_phase([sys.executable, "-c", child], probe / "timeout.log", 1, os.environ.copy())
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError("Timeout accepted")
    try:
        os.kill(int(pid.read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("Timed-out process survived")
    # The output mount is the platform's capture boundary; preserve failed-step
    # logs and never clean it in the wrapper. Rehearse a capture failure locally.
    assert (probe / "failure.log").exists()
    try:
        with (probe / "failure.log" / "invalid-output").open("w"):
            pass
    except OSError:
        pass
    else:
        raise AssertionError("Expected simulated capture failure")
    assert "failure evidence" in (probe / "failure.log").read_text()
    record = dict(
        source_revision=package["source_revision"],
        archive_sha256=package["archive_sha256"],
        binary_sha256=built.binary_sha256,
        geometry=pilot.training_geometry(),
        migration=prepared.initialization.rollout_migration,
        dependency_versions=actual,
        phase_success=True,
        phase_failure_preserved=True,
        timeout_child_reaped=True,
        capture_failure_preserves_evidence=True,
        scope="CPU only: exact image input/build/resume and owned-phase lifecycle; GPU/strength remain unqualified",
    )
    (kit / "cpu-ready.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.copytree(probe, kit / "cpu-lifecycle-evidence")
    for name in ("config.json", "build-config.json", "continuation.json"):
        shutil.copy2(work / "out" / name, kit / ("cpu-" + name))
    # Only this proof's newly created duplicate scratch is removed; original inputs,
    # compiled source, checkpoint archive, configs and lifecycle evidence stay in image.
    Path("/recovery").unlink()
    for child in work.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    print("H100_JOB_IMAGE_CPU_READY", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

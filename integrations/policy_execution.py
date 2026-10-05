"""Finite policy subprocesses with owned shutdown and measured training gates."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def runtime_environment(source, output, sampler):
    env = dict(os.environ)
    # Optional sampler fields must never inherit a previous experiment's settings.
    for key in (
        "POLICY_TEMPERATURE",
        "SPLIT_TEMPERATURE",
        "EARLY_ROUTE_TEMPERATURE",
        "EARLY_ROUTE_TURNS",
        "FULL_ACTION_TEMPERATURE",
        "LOG_GAP_SCALE",
        "ROUTE_HALF_WEIGHT",
        "NEUTRAL_ROUTE_BIAS",
        "WEAK_OWNED_ROUTE_PENALTY",
        "DOOMED_ATTACK_ROUTE_PENALTY",
    ):
        env.pop("METTA_SPATIAL_" + key, None)
    env.update(
        PYTHONPATH=f"{source}/integrations/puffer_bootstrap:{source}:/opt/generals-source",
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        TMPDIR="/work/tmp",
        XDG_CACHE_HOME="/work/cache",
        JAX_COMPILATION_CACHE_DIR="/work/jax-cache",
        FABRIC_VERIFY_CACHE="/work/fabric-verify",
        METTA_PUFFER_SOURCE_REPOSITORY="/work/input/puffer.git",
        METTA_PUFFER_RAYLIB_DIRECTORY="/work/input/raylib-5.5_linux_amd64",
        METTA_MEMORYLESS_OPTIMIZATION="1",
        METTA_DIRECT_SPATIAL_ROLLOUT="1",
        METTA_SPATIAL_MUON_DENSE_ORIENTATION="canonical",
        METTA_SPATIAL_MUON_CONTEXT_MATRIX="1",
        METTA_SPATIAL_OPTIMIZER_LAYOUT="logical",
        METTA_AUDIT_DEVICE_REWARDS="1",
        METTA_AUDIT_SPATIAL_SPLITS="1",
        METTA_AUDIT_ACTION_MASK="1",
        METTA_AUDIT_POPULATION_WINS="1",
        METTA_EPOCH_TIMING_DIR=str(output / "timing"),
        METTA_SPATIAL_SAMPLING_GATE_REPORT=str(output / "sampling-gate.json"),
    )
    for name, value in sampler.items():
        if name == "mode":
            continue
        key = (
            "POLICY_TEMPERATURE"
            if name == "move_temperature"
            else "SPLIT_TEMPERATURE"
            if name == "split_temperature"
            else name.upper()
        )
        if value is not None:
            env["METTA_SPATIAL_" + key] = str(value)
    return env


def execute(module, arguments, *, source, output, sampler, name, seconds, training_config=None):
    """Start one process group; preserve logs on every outcome."""
    output.mkdir(parents=True, exist_ok=True)
    env = runtime_environment(source, output, sampler)
    if name == "preflight":
        env.update(JAX_PLATFORMS="cpu", METTA_AUDIT_DEVICE_REWARDS="0")
    steps_per_epoch = None
    if training_config:
        config = json.loads(Path(training_config).read_text())["overrides"]
        steps_per_epoch = config["vec.total_agents"] * config["train.horizon"]

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (output / f"{name}.log").open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "integrations." + module, *map(str, arguments)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started, sampled = time.monotonic(), 0.0
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > seconds:
                        raise TimeoutError(f"{name} exceeded {seconds}s; inspect retained log")
                    if training_config:
                        from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps
                        from integrations.slurm_s3_job import allocated_gpu_identity

                        if elapsed - sampled >= 5:
                            with (output / "gpu.csv").open("ab") as samples:
                                subprocess.run(
                                    [
                                        "nvidia-smi",
                                        "--id=" + allocated_gpu_identity()["uuid"],
                                        "--query-gpu=timestamp,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader",
                                    ],
                                    stdout=samples,
                                    stderr=log,
                                    check=True,
                                    timeout=10,
                                )
                            sampled = elapsed
                        console = output / "run/console.log"
                        text = console.read_text(errors="replace") if console.exists() else ""
                        if "NonFiniteGradsError" in text or "FloatingPointError" in text:
                            raise FloatingPointError("Native training produced nonfinite values")
                        times = completed_epoch_times(text)
                        if elapsed > 300 and not times:
                            raise TimeoutError("No completed training epoch after 300s")
                        sps = interval_sps(times, 2, steps_per_epoch)
                        if len(times) >= 4 and sps is not None and sps < 30_000:
                            raise RuntimeError(f"Sustained training throughput below 30,000 SPS: {sps}")
                    time.sleep(1)
                if process.returncode:
                    raise RuntimeError(f"{name} exited {process.returncode}; inspect retained log")
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def training_audit(output, config):
    from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps

    completed = json.loads((output / "run/completed.json").read_text())
    expected = config["total_timesteps"]
    if completed["trained_timesteps"] != expected:
        raise ValueError("Incomplete native training budget")
    text = (output / "run/console.log").read_text()
    if f"DEVICE_ACTION_MASK_AUDIT actions={expected} illegal=0" not in text:
        raise ValueError("Complete zero-illegal-action audit is missing")
    rewards = [
        json.loads(line.split("DEVICE_REWARD_AUDIT ", 1)[1])
        for line in text.splitlines()
        if line.startswith("DEVICE_REWARD_AUDIT ")
    ]
    if (
        not rewards
        or rewards[-1]["agent_steps"] != expected
        or any(
            rewards[-1][key]
            for key in ("nonfinite_rewards", "native_clipped_rewards", "native_clipped_terminal_rewards")
        )
    ):
        raise ValueError("Complete finite reward audit is missing")
    population = json.loads(
        (output / f"run/environments/{config['seed']}/spatial-opponent-population.json").read_text()
    )
    build = json.loads((output.parent / "build-config.json").read_text())
    pool = build["python_environment"]["options"]
    expected_hashes = [
        hashlib.sha256((Path(path) / "policy.bin").read_bytes()).hexdigest() for path in pool["frozen_bundles"]
    ]
    expected_names = {"frozen_" + digest[:12] for digest in expected_hashes} | set(pool["scripted_opponents"])
    if (
        population["frozen_policy_sha256"] != expected_hashes
        or population["opponent_weights"] != pool["opponent_weights"]
        or set(population["counts"]) != expected_names
        or not all(counts["0"] > 0 and counts["0"] == counts["1"] for counts in population["counts"].values())
    ):
        raise ValueError("Opponent identity or balanced seat allocation differs")
    options = config["overrides"]
    batch = options["vec.total_agents"] * options["train.horizon"]
    epochs = completed_epoch_times(text)
    sps = interval_sps(epochs, 2, batch)
    if sps is None or sps < 30_000:
        raise ValueError("Completed run does not qualify 30,000 end-to-end SPS")
    report = {
        "environment_steps": expected,
        "environment_count": options["vec.total_agents"],
        "horizon": options["train.horizon"],
        "minibatch": options["train.minibatch_size"],
        "replay_ratio": options["train.replay_ratio"],
        "steady_sps": sps,
        "epoch_uptime": epochs,
        "reward_audit": rewards[-1],
        "illegal_actions": 0,
        "opponent_counts_by_seat": population["counts"],
    }
    (output / "training-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

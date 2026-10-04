"""Bounded Classic pilot entry points for the immutable Pyxis input image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PARENT = Path("/recovery/classic-split-pool-67m-35633")
CHECKPOINT_SHA = "3eb2fe3f22affcc4f7563c46c105a2d71e183b73447c3e6e617b1a4da7c0aeeb"
SOURCE = Path("/work/input/source")
OUT = Path("/work/out")


def pilot_steps(value):
    steps = int(value)
    if steps not in (8_388_608, 33_554_432, 268_435_456):
        raise ValueError("Budget must be 8M qualification, 32M comparison, or 256M learning curve")
    return steps


STEPS = pilot_steps(os.environ.get("GENERALS_PILOT_STEPS", "8388608"))


def configure_reward_scale(options, value):
    """Keep the duration control reproducible; explicitly opt into unclipped rewards."""
    scale = float(value)
    if scale not in (1.0, 0.5):
        raise ValueError("Reward-scale comparison permits only 1 or 0.5")
    expected = dict(
        terminal_reward_mode="win_only",
        shaping_weight=0.25,
        shaping_gamma=0.999,
        army_shaping_weight=0.5,
        land_shaping_weight=0.3,
        castle_shaping_weight=0.0,
        imitation_weight=0.0,
        land_gain_reward_weight=0.0,
    )
    if any(options.get(key) != val for key, val in expected.items()) or options.get("frontier_shaping_weight", 0.0):
        raise ValueError("Reward bound requires the exact Classic potential objective")
    if options.get("reward_scale") != 1.0:
        raise ValueError("Expected the unscaled parent reward contract")
    options["reward_scale"] = scale
    # Each army/land margin lies in [-1, 1]. Terminal potential is zero.
    potential_bound = 0.25 * (0.5 + 0.3)
    return scale * max(1 + potential_bound, (1 + 0.999) * potential_bound)


TRAIN_ENV = dict(
    METTA_SPATIAL_MUON_DENSE_ORIENTATION="canonical",
    METTA_SPATIAL_MUON_CONTEXT_MATRIX="1",
    METTA_SPATIAL_OPTIMIZER_LAYOUT="logical",
    METTA_SPATIAL_POLICY_TEMPERATURE="0.05",
    METTA_SPATIAL_SPLIT_TEMPERATURE="0.15",
    METTA_SPATIAL_FULL_ACTION_TEMPERATURE=os.environ.get("GENERALS_FULL_ACTION_TEMPERATURE", "1"),
    METTA_SPATIAL_LOG_GAP_SCALE=os.environ.get("GENERALS_LOG_GAP_SCALE", "0"),
    METTA_SPATIAL_ROUTE_HALF_WEIGHT="0.0",
    METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE="0.10",
    METTA_SPATIAL_EARLY_ROUTE_TURNS="100",
    METTA_SPATIAL_NEUTRAL_ROUTE_BIAS="6.0",
    METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY="4.0",
    METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY="4.0",
    METTA_DIRECT_SPATIAL_ROLLOUT="1",
    METTA_MEMORYLESS_OPTIMIZATION="1",
    METTA_AUDIT_DEVICE_REWARDS="1",
    METTA_AUDIT_SPATIAL_SPLITS="1",
    METTA_AUDIT_ACTION_MASK="1",
    METTA_AUDIT_POPULATION_WINS="1",
    METTA_SPATIAL_SAMPLING_GATE_REPORT=str(OUT / "sampling-gate.json"),
    METTA_EPOCH_TIMING_DIR=str(OUT / "timing"),
)


def training_geometry():
    """Read the actual rollout dimensions; never inflate SPS using a parent batch."""
    overrides = json.loads((OUT / "config.json").read_text())["overrides"]
    names = {"parallel_games": "vec.total_agents", "horizon": "train.horizon",
             "minibatch": "train.minibatch_size"}
    values = {name: overrides[key] for name, key in names.items()}
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values.values()):
        raise ValueError("Training geometry requires positive integer dimensions")
    values["steps_per_epoch"] = values["parallel_games"] * values["horizon"]
    if values["steps_per_epoch"] % values["minibatch"]:
        raise ValueError("Training minibatch must divide the complete rollout")
    return values


def allocated_gpu():
    from integrations.slurm_s3_job import allocated_gpu_identity

    return allocated_gpu_identity()["uuid"]


def continuation_parent():
    if (OUT / "context-parent/migration.json").is_file():
        return OUT / "context-parent"
    path = os.environ.get("GENERALS_PILOT_CONTINUATION_MANIFEST")
    return Path(path).parent / "parent" if path else PARENT


def starting_steps():
    path = os.environ.get("GENERALS_PILOT_CONTINUATION_MANIFEST")
    return json.loads(Path(path).read_text())["agent_steps"] if path else 0


def prepare_configs(parent=PARENT, output=OUT):
    continuation = os.environ.get("GENERALS_PILOT_CONTINUATION_MANIFEST")
    if continuation:
        from integrations.classic_learner_continuation import load_continuation, verify_factory_source

        verify_factory_source(Path(continuation).parent / "parent", Path(__file__).with_name("generals_fabric.py"))
        parent, build, run, resume = load_continuation(continuation, STEPS)
        options = build["python_environment"]["options"]
        # Validate the same potential objective without scaling it twice.
        unscaled = dict(options, reward_scale=1.0)
        bound = configure_reward_scale(unscaled, options["reward_scale"])
        from integrations.environment_reward_audit import population_win_threshold

        threshold = population_win_threshold(options["terminal_reward_mode"], options)
        (output / "reward-contract.json").write_text(
            json.dumps(
                dict(
                    reward_scale=options["reward_scale"],
                    absolute_reward_bound=bound,
                    native_clamp=[-1, 1],
                    expected_unclipped=bound <= 1,
                    population_win_threshold=threshold,
                ),
                indent=2,
            )
            + "\n"
        )
        position_manifest = os.environ.get("GENERALS_PILOT_POSITION_MANIFEST")
        if position_manifest:
            from integrations.classic_position_curriculum import configure_positions

            curriculum = configure_positions(options, position_manifest)
            (output / "position-curriculum.json").write_text(json.dumps(curriculum, indent=2) + "\n")
        if resume["context_extension"] == "preserve":
            shutil.copytree(parent / "bundle", output / "self_bundle")
        for name, value in (("build-config.json", build), ("config.json", run), ("continuation.json", resume)):
            (output / name).write_text(json.dumps(value, indent=2) + "\n")
        return build, run
    build = json.loads((parent / "build-config.json").read_text())
    run = json.loads((parent / "config.json").read_text())
    source_checkpoint = parent / "run/checkpoints/metta_generals/run/0000000067108864.bin"
    if hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Parent checkpoint differs")
    options = build["python_environment"]["options"]
    if (
        not options["coworld_classic"]
        or options["terminal_reward_mode"] != "win_only"
        or options["shaping_gamma"] != run["overrides"]["train.gamma"]
        or options["teacher"] is not None
        or options["teacher_rollouts"]
    ):
        raise ValueError("Classic rules, reward discount or teacher settings differ")
    if len(options["frozen_bundles"]) != 9:
        raise ValueError("Parent opponent pool differs")
    bound = configure_reward_scale(options, os.environ.get("GENERALS_PILOT_REWARD_SCALE", "1"))
    from integrations.environment_reward_audit import population_win_threshold

    threshold = population_win_threshold(options["terminal_reward_mode"], options)
    (output / "reward-contract.json").write_text(
        json.dumps(
            dict(
                reward_scale=options["reward_scale"],
                absolute_reward_bound=bound,
                native_clamp=[-1, 1],
                expected_unclipped=bound <= 1,
                population_win_threshold=threshold,
            ),
            indent=2,
        )
        + "\n"
    )
    position_manifest = os.environ.get("GENERALS_PILOT_POSITION_MANIFEST")
    if position_manifest:
        from integrations.classic_position_curriculum import configure_positions

        curriculum = configure_positions(options, position_manifest)
        (output / "position-curriculum.json").write_text(json.dumps(curriculum, indent=2) + "\n")
    shutil.copytree(parent / "bundle", output / "self_bundle")
    options["frozen_bundles"].append(str(output / "self_bundle"))
    options["opponent_weights"] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 6]
    options["coworld_pool_size"] = 8192
    run.update(total_timesteps=STEPS, seed=8842)
    run["overrides"].update({"train.learning_rate": 0.0002, "train.ent_coef": 0.0})
    run["initialize"].update(
        run=str(parent / "run"), checkpoint=str(source_checkpoint), sha256=CHECKPOINT_SHA, restore_learner=False
    )
    for name, value in (("build-config.json", build), ("config.json", run)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return build, run


def command(module, *args, name, seconds=600, train=False):
    env = dict(
        os.environ,
        METTA_SPATIAL_MUON_DENSE_ORIENTATION="canonical",
        METTA_SPATIAL_MUON_CONTEXT_MATRIX="1",
        METTA_SPATIAL_OPTIMIZER_LAYOUT="logical",
    )
    if train:
        env.update(TRAIN_ENV)
    argv = [sys.executable, "-m", "integrations." + module, *map(str, args)]
    with (OUT / (name + ".log")).open("wb") as log:
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        sampled = 0.0
        while process.poll() is None:
            if time.monotonic() - started > seconds:
                process.terminate()
                # The outer srun owns the whole process tree and independently
                # enforces its bounded step timeout before archive/upload.
                raise RuntimeError(f"{name} exceeded its runtime budget")
            if train:
                if time.monotonic() - sampled >= 5:
                    with (OUT / "gpu.csv").open("ab") as samples:
                        subprocess.run(
                            [
                                "nvidia-smi",
                                "--id=" + allocated_gpu(),
                                "--query-gpu=timestamp,uuid,memory.used,utilization.gpu",
                                "--format=csv,noheader",
                            ],
                            stdout=samples,
                            stderr=log,
                            timeout=3,
                            check=True,
                        )
                    sampled = time.monotonic()
                console = OUT / "run/console.log"
                if console.exists():
                    text = console.read_text(errors="replace")
                    if (
                        "pooling:" in text
                        and "SPATIAL_ADAPTER_ACTIVE module=integrations.direct_spatial_optimization" not in text
                    ):
                        process.terminate()
                        raise RuntimeError("Native worker entered Fabric setup without the verified spatial adapter")
                    if "NonFiniteGradsError" in text or "FloatingPointError" in text:
                        process.terminate()
                        raise RuntimeError("Nonfinite training result")
                    from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps

                    times = completed_epoch_times(text)
                    sps = interval_sps(times, span=2, steps_per_epoch=training_geometry()["steps_per_epoch"])
                    if max(times, default=0) >= 3 and sps is not None and sps < 30_000:
                        process.terminate()
                        raise RuntimeError("Sustained end-to-end training SPS below 30000")
            time.sleep(1)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, [name])


def smoke():
    from integrations.slurm_s3_job import verify_allocated_gpu_idle

    identity = verify_allocated_gpu_idle()
    (OUT / "gpu-preflight.json").write_text(json.dumps(identity, indent=2) + "\n")
    import jax

    if len(jax.devices("gpu")) != 1:
        raise RuntimeError("Expected exactly one allocated JAX GPU")
    subprocess.run(
        [
            "nvidia-smi",
            "--id=" + allocated_gpu(),
            "--query-gpu=uuid,name,driver_version,memory.total,memory.used",
            "--format=csv",
        ],
        check=True,
        timeout=15,
    )
    manifest = json.loads(Path("/work/input/source-manifest.json").read_text())
    for relative, digest in manifest.items():
        if hashlib.sha256((Path("/work/input") / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Input source or checkpoint hash differs: " + relative)
    prepare_configs()
    print("SMOKE_OK", jax.__version__, flush=True)


def build():
    command(
        "launch_spatial_selfplay_training",
        "build",
        "--config",
        OUT / "build-config.json",
        "--output",
        OUT / "build",
        name="build-native",
        seconds=540,
    )
    manifest = json.loads((OUT / "build/build.json").read_text())
    from integrations.materialize_spatial_context_extension import EXTENDED_MODEL, PARENT_MODEL, materialize

    radius = manifest["config"]["fabric"]["options"]["context_radius"]
    if radius not in (1.01, 2.01) or manifest["model_sha256"] != (
        EXTENDED_MODEL if radius == 2.01 else PARENT_MODEL
    ):
        raise ValueError("Native model fingerprint changed during image migration")
    continuation = OUT / "continuation.json"
    if continuation.is_file() and json.loads(continuation.read_text()).get("context_extension") == "zero_extend_radius2":
        reference = materialize(continuation_parent(), OUT / "build", OUT / "context-parent",
                                SOURCE / "integrations/generals_fabric.py")
        run_path = OUT / "config.json"
        run = json.loads(run_path.read_text())
        run["initialize"] = reference
        run_path.write_text(json.dumps(run, indent=2) + "\n")
        shutil.copytree(OUT / "context-parent/bundle", OUT / "self_bundle")


def full_action_temperature():
    from integrations.spatial_action_sampling import validate_full_action_temperature

    return validate_full_action_temperature(float(TRAIN_ENV["METTA_SPATIAL_FULL_ACTION_TEMPERATURE"]))


def log_gap_scale():
    from integrations.spatial_exploration import validate_log_gap_scale

    return validate_log_gap_scale(float(TRAIN_ENV["METTA_SPATIAL_LOG_GAP_SCALE"]))


def exploratory_initialization():
    """Copy the qualified parent with the exact acting sampler, retaining the cold parent."""
    from integrations.spatial_policy_bundle import SpatialPlayerPolicy

    source, target = OUT / "self_bundle", OUT / "exploratory-initialization"
    SpatialPlayerPolicy(source)  # Verify source payloads before copying anything.
    temperature = full_action_temperature()
    shutil.copytree(source, target)
    path = target / "spatial-policy.json"
    record = json.loads(path.read_text())
    if record["serving_action_selection"]["mode"] != "structured_sample":
        raise ValueError("Exploration requires a structured parent policy")
    record["serving_action_selection"]["full_action_temperature"] = temperature
    record["serving_action_selection"]["log_gap_scale"] = log_gap_scale()
    path.write_text(json.dumps(record, indent=2) + "\n")
    policy = SpatialPlayerPolicy(target)
    if policy.full_action_temperature != temperature or policy.log_gap_scale != log_gap_scale():
        raise ValueError("Exploratory initialization sampler differs")
    return target


def sampling_gate():
    continuing = bool(starting_steps())
    # A resumed policy already uses the scheduled sampler. Its viability check
    # must not require winning with the obsolete unscheduled ablation.
    for label in ("candidate",) if continuing else ("baseline", "candidate"):
        extra = ["--early-route-temperature", ".10", "--early-route-turns", "100"] if label == "candidate" else []
        command(
            "evaluate_spatial_frozen_match",
            "--bundle",
            OUT / "self_bundle",
            "--run",
            continuation_parent() / "run",
            "--opponent-bundle",
            OUT / "self_bundle",
            "--opponent-run",
            continuation_parent() / "run",
            "--games",
            512,
            "--pool-size",
            512,
            "--seed",
            35821,
            "--sample-seed",
            31337,
            "--sampling-temperature",
            0.05,
            "--split-sampling-temperature",
            0.15,
            "--full-action-temperature",
            full_action_temperature() if label == "candidate" else 1.0,
            "--log-gap-scale",
            log_gap_scale() if label == "candidate" else 0.0,
            "--neutral-route-bias",
            6,
            "--weak-owned-route-penalty",
            4,
            "--doomed-attack-route-penalty",
            4,
            *extra,
            "--output",
            OUT / "gate" / label,
            name="gate-" + label,
            seconds=360,
        )
    command(
        "analyze_spatial_frozen_match_pair",
        "--baseline",
        OUT / ("gate/candidate" if continuing else "gate/baseline"),
        "--candidate",
        OUT / "gate/candidate",
        "--allow-policy-mode-change",
        "--seed",
        35623,
        "--output",
        OUT / "sampling-gate.json",
        name="gate-pair",
        seconds=60,
    )
    report = json.loads((OUT / "sampling-gate.json").read_text())
    if continuing:
        report["gate_mode"] = "same_sampler_continuation"
        report["scope"] = "One 512-game self-match under the resumed sampler; no improvement comparison"
        (OUT / "sampling-gate.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["score_delta"] < -0.12 or report["candidate_wld"][0] < 128:
        raise ValueError("Parent policy fails the training sampler gate")


def train():
    (OUT / "timing").mkdir()
    command(
        "launch_spatial_selfplay_training",
        "train",
        "--config",
        OUT / "config.json",
        "--build",
        OUT / "build",
        "--output",
        OUT / "run",
        name="train-native",
        # The qualified native-siege pool sustains ~72k environment SPS;
        # 268M steps need ~62 minutes before compilation/checkpoint overhead.
        # The host step and finite Slurm allocation remain the outer limits.
        # The 33M architecture pilot must accommodate the 30k SPS floor plus
        # bounded warmup, rather than implicitly requiring the parent's 85k.
        seconds=4440 if STEPS == 268_435_456 else (1440 if STEPS == 33_554_432 else 840),
        train=True,
    )
    if not (OUT / "run/completed.json").is_file():
        raise ValueError("Trainer did not finish and validate its checkpoints")
    text = (OUT / "run/console.log").read_text()
    if f"DEVICE_ACTION_MASK_AUDIT actions={STEPS} illegal=0" not in text:
        raise ValueError("Complete legal-action audit missing")
    audits = [
        json.loads(line.split("DEVICE_REWARD_AUDIT ", 1)[1])
        for line in text.splitlines()
        if line.startswith("DEVICE_REWARD_AUDIT ")
    ]
    if not audits or audits[-1]["agent_steps"] != STEPS or audits[-1]["nonfinite_rewards"]:
        raise ValueError("Reward audit missing or nonfinite")
    reward_contract = json.loads((OUT / "reward-contract.json").read_text())
    if reward_contract["expected_unclipped"] and audits[-1]["native_clipped_rewards"]:
        raise ValueError("Scaled reward violates the audited native clamp bound")
    population = json.loads((OUT / "run/environments/8842/spatial-opponent-population.json").read_text())
    expected_pool = json.loads((OUT / "build-config.json").read_text())["python_environment"]["options"]
    expected_hashes = [
        hashlib.sha256((Path(bundle) / "policy.bin").read_bytes()).hexdigest()
        for bundle in expected_pool["frozen_bundles"]
    ]
    expected_scripts = expected_pool.get("scripted_opponents", ["expander_harvester", "sentinel"])
    expected_names = {"frozen_" + digest[:12] for digest in expected_hashes} | set(expected_scripts)
    if (
        len(population["frozen_action_selection"]) != 10
        or population["opponent_weights"] != expected_pool["opponent_weights"]
        or population["frozen_policy_sha256"] != expected_hashes
        or set(population["counts"]) != expected_names
        or not all(c["0"] > 0 and c["0"] == c["1"] for c in population["counts"].values())
    ):
        raise ValueError("Opponent pool or balanced seat allocation differs")
    if "classic_siege_padded" in expected_scripts:
        from integrations.classic_siege_native import SOURCE

        native = population.get("native_opponent", {})
        if (native.get("name") != "classic_siege_padded"
                or native.get("workers") != expected_pool.get("classic_siege_workers", 1)
                or native.get("source_sha256") != hashlib.sha256(SOURCE.read_bytes()).hexdigest()):
            raise ValueError("Native siege opponent source binding differs")
    from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps

    times = completed_epoch_times(text)
    sps = interval_sps(times, span=2, steps_per_epoch=training_geometry()["steps_per_epoch"])
    if sps is None or sps < 30_000:
        raise ValueError("Measured end-to-end training interval failed the SPS gate")
    (OUT / "training-audit.json").write_text(
        json.dumps(
            dict(
                environment_steps=STEPS,
                starting_agent_steps=starting_steps(),
                ending_agent_steps=starting_steps() + STEPS,
                **training_geometry(),
                replay_ratio=0.5,
                steady_sps=sps,
                epoch_uptime=times,
                reward_audit=audits[-1],
            ),
            indent=2,
        )
        + "\n"
    )


def export_and_audit(steps, bundle, suffix=""):
    checkpoint = OUT / f"run/checkpoints/metta_generals/run/{steps:016d}.bin"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    command(
        "export_spatial_policy_bundle",
        "--build",
        OUT / "build/build.json",
        "--training",
        OUT / "run/training.json",
        "--checkpoint",
        checkpoint,
        "--sha256",
        digest,
        "--factory-source",
        SOURCE / "integrations/generals_fabric.py",
        "--output",
        bundle,
        "--serving-move-temperature",
        0.05,
        "--serving-split-temperature",
        0.15,
        "--serving-full-action-temperature",
        full_action_temperature(),
        "--serving-log-gap-scale",
        log_gap_scale(),
        "--serving-early-route-temperature",
        0.1,
        "--serving-early-route-turns",
        100,
        "--serving-neutral-route-bias",
        6,
        "--serving-weak-owned-route-penalty",
        4,
        "--serving-doomed-attack-route-penalty",
        4,
        name="export" + suffix,
        seconds=180,
    )
    report_path = OUT / ("checkpoint-serving-parity" + suffix + ".json")
    command(
        "audit_spatial_checkpoint_serving_parity",
        "--bundle",
        bundle,
        "--replay-root",
        "/work/input/leader-root",
        "--factory-source",
        SOURCE / "integrations/generals_fabric.py",
        "--output",
        report_path,
        name="serving-parity" + suffix,
        seconds=300,
    )
    report = json.loads(report_path.read_text())
    if (
        report["checkpoint_sha256"] != digest
        or report["public_states"] < 40
        or report["matching_top_actions"] != report["public_states"]
        or report["max_action_probability_difference"] > 1e-5
        or report["max_logit_difference"] > 2e-5
        or report["max_rollout_transform_difference"] > 1e-5
    ):
        raise ValueError("Trained checkpoint failed serving parity")


def midpoint_checkpoint_steps(start, additional_steps):
    """Choose a retained interior checkpoint, recording any epoch-grid offset."""
    requested = start + additional_steps // 2
    directory = OUT / "run/checkpoints/metta_generals/run"
    candidates = [int(path.stem) for path in directory.glob("*.bin")
                  if path.stem.isdigit() and start < int(path.stem) < start + additional_steps]
    if not candidates:
        raise FileNotFoundError("No retained interior checkpoint for the learning curve")
    selected = min(candidates, key=lambda step: (abs(step - requested), step))
    config = json.loads((OUT / "config.json").read_text())
    interval = config["overrides"]["base.checkpoint_interval"]
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("Expected a positive checkpoint epoch interval")
    if abs(selected - requested) > interval * training_geometry()["steps_per_epoch"] // 2:
        raise FileNotFoundError("No retained checkpoint within half a save interval of midpoint")
    (OUT / "midpoint-selection.json").write_text(json.dumps(dict(
        requested_agent_steps=requested, selected_agent_steps=selected,
        offset_steps=selected - requested, checkpoint_interval_epochs=interval,
        selection="Nearest retained interior checkpoint; ties choose the earlier checkpoint",
    ), indent=2) + "\n")
    return selected


def evaluation_seeds():
    """Allow a fresh held-out panel when branching again from the same parent."""
    overrides = [os.environ.get(key) for key in ("GENERALS_EVAL_SEED", "GENERALS_EVAL_SAMPLE_SEED")]
    if any(value is not None for value in overrides):
        if any(value is None for value in overrides):
            raise ValueError("Evaluation map and sampling seed overrides must be paired")
        seeds = tuple(int(value) for value in overrides)
        if any(not 0 <= value < 2**32 for value in seeds):
            raise ValueError("Evaluation seed overrides must be uint32 values")
        return seeds
    # Keep the extended run off the repeatedly used short-pilot development maps.
    map_seed, sample_seed = (37841, 8729) if STEPS == 268_435_456 else (37813, 8713)
    # Distinct checkpoint counters, including a chosen midpoint, get new maps.
    if starting_steps():
        map_seed, sample_seed = 37871 + starting_steps() // (8192 * 256), 8753 + starting_steps() // (8192 * 256)
    return map_seed, sample_seed


def evaluate():
    map_seed, sample_seed = evaluation_seeds()
    export_and_audit(starting_steps() + STEPS, OUT / "bundle")
    arms = [("parent", OUT / "self_bundle"), ("child", OUT / "bundle")]
    if STEPS >= 33_554_432:
        export_and_audit(midpoint_checkpoint_steps(starting_steps(), STEPS), OUT / "bundle-mid", "-mid")
        arms.append(("mid", OUT / "bundle-mid"))
    if full_action_temperature() != 1 or log_gap_scale():
        # Improvement over a hotter initialization alone is insufficient: retain
        # an independent matched panel against the original qualified parent.
        arms[0] = ("parent", exploratory_initialization())
        arms.append(("cold-parent", OUT / "self_bundle"))
    for label, bundle in arms:
        command(
            "evaluate_spatial_population",
            "--bundle",
            bundle,
            "--population-build",
            OUT / "build/build.json",
            "--games",
            4096,
            "--pool-size",
            4096,
            "--seed",
            map_seed,
            "--sample-seed",
            sample_seed,
            "--output",
            OUT / ("heldout-" + label),
            name="heldout-" + label,
            seconds=360,
        )
    comparisons = [("parent", label, "paired-" + label) for label, _ in arms[1:]]
    if full_action_temperature() != 1 or log_gap_scale():
        comparisons += [("cold-parent", label, "paired-cold-" + label)
                        for label, _ in arms if label in ("child", "mid")]
    for baseline, label, report_name in comparisons:
        command(
            "analyze_spatial_population_pair",
            "--baseline",
            OUT / ("heldout-" + baseline),
            "--candidate",
            OUT / ("heldout-" + label),
            "--output",
            OUT / (report_name + ".json"),
            name=report_name,
            seconds=60,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "build", "sampling_gate", "train", "evaluate"))
    args = parser.parse_args()
    from integrations.cuda_runtime_binding import audit, configure

    configure()
    (OUT / ("cuda-binding-" + args.phase + ".json")).write_text(json.dumps(audit(), indent=2) + "\n")
    os.environ.update(
        PYTHONPATH=f"{SOURCE}/integrations/puffer_bootstrap:{SOURCE}:/opt/generals-source",
        JAX_PLATFORMS="cuda,cpu",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XDG_CACHE_HOME="/work/cache",
        FABRIC_VERIFY_CACHE="/work/fabric-verify",
        JAX_COMPILATION_CACHE_DIR="/work/jax-cache",
        TMPDIR="/work/tmp",
        METTA_PUFFER_SOURCE_REPOSITORY="/work/input/puffer.git",
        METTA_PUFFER_RAYLIB_DIRECTORY="/work/input/raylib-5.5_linux_amd64",
    )
    Path("/work/tmp").mkdir(exist_ok=True)
    globals()[args.phase]()


if __name__ == "__main__":
    main()

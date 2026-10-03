"""Bounded Classic pilot entry points for the immutable Pyxis input image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


PARENT = Path("/recovery/classic-split-pool-67m-35633")
CHECKPOINT_SHA = "3eb2fe3f22affcc4f7563c46c105a2d71e183b73447c3e6e617b1a4da7c0aeeb"
SOURCE = Path("/work/input/source")
OUT = Path("/work/out")
def pilot_steps(value):
    steps = int(value)
    if steps not in (8_388_608, 33_554_432):
        raise ValueError("Pilot budget must be 8M qualification or 32M learning curve")
    return steps


STEPS = pilot_steps(os.environ.get("GENERALS_PILOT_STEPS", "8388608"))


def configure_reward_scale(options, value):
    """Keep the duration control reproducible; explicitly opt into unclipped rewards."""
    scale = float(value)
    if scale not in (1.0, 0.5):
        raise ValueError("Reward-scale comparison permits only 1 or 0.5")
    expected = dict(terminal_reward_mode="win_only", shaping_weight=.25,
                    shaping_gamma=.999, army_shaping_weight=.5,
                    land_shaping_weight=.3, castle_shaping_weight=0.,
                    imitation_weight=0., land_gain_reward_weight=0.)
    if any(options.get(key) != val for key, val in expected.items()) or options.get("frontier_shaping_weight", 0.):
        raise ValueError("Reward bound requires the exact Classic potential objective")
    if options.get("reward_scale") != 1.0:
        raise ValueError("Expected the unscaled parent reward contract")
    options["reward_scale"] = scale
    # Each army/land margin lies in [-1, 1]. Terminal potential is zero.
    potential_bound = .25 * (.5 + .3)
    return scale * max(1 + potential_bound, (1 + .999) * potential_bound)


TRAIN_ENV = dict(
    METTA_SPATIAL_MUON_DENSE_ORIENTATION="canonical", METTA_SPATIAL_MUON_CONTEXT_MATRIX="1",
    METTA_SPATIAL_OPTIMIZER_LAYOUT="logical", METTA_SPATIAL_POLICY_TEMPERATURE="0.05",
    METTA_SPATIAL_SPLIT_TEMPERATURE="0.15", METTA_SPATIAL_ROUTE_HALF_WEIGHT="0.0",
    METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE="0.10", METTA_SPATIAL_EARLY_ROUTE_TURNS="100",
    METTA_SPATIAL_NEUTRAL_ROUTE_BIAS="6.0", METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY="4.0",
    METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY="4.0", METTA_DIRECT_SPATIAL_ROLLOUT="1",
    METTA_MEMORYLESS_OPTIMIZATION="1", METTA_AUDIT_DEVICE_REWARDS="1",
    METTA_AUDIT_SPATIAL_SPLITS="1", METTA_AUDIT_ACTION_MASK="1", METTA_AUDIT_POPULATION_WINS="1",
    METTA_SPATIAL_SAMPLING_GATE_REPORT=str(OUT / "sampling-gate.json"),
    METTA_EPOCH_TIMING_DIR=str(OUT / "timing"),
)


def allocated_gpu():
    from integrations.slurm_s3_job import allocated_gpu_identity
    return allocated_gpu_identity()["uuid"]


def prepare_configs(parent=PARENT, output=OUT):
    build = json.loads((parent / "build-config.json").read_text())
    run = json.loads((parent / "config.json").read_text())
    source_checkpoint = parent / "run/checkpoints/metta_generals/run/0000000067108864.bin"
    if hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Parent checkpoint differs")
    options = build["python_environment"]["options"]
    if (not options["coworld_classic"] or options["terminal_reward_mode"] != "win_only"
            or options["shaping_gamma"] != run["overrides"]["train.gamma"]
            or options["teacher"] is not None or options["teacher_rollouts"]):
        raise ValueError("Classic rules, reward discount or teacher settings differ")
    if len(options["frozen_bundles"]) != 9:
        raise ValueError("Parent opponent pool differs")
    bound = configure_reward_scale(options, os.environ.get("GENERALS_PILOT_REWARD_SCALE", "1"))
    (output / "reward-contract.json").write_text(json.dumps(dict(
        reward_scale=options["reward_scale"], absolute_reward_bound=bound,
        native_clamp=[-1, 1], expected_unclipped=bound <= 1,
    ), indent=2) + "\n")
    shutil.copytree(parent / "bundle", output / "self_bundle")
    options["frozen_bundles"].append(str(output / "self_bundle"))
    options["opponent_weights"] = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 6]
    options["coworld_pool_size"] = 8192
    run.update(total_timesteps=STEPS, seed=8842)
    run["overrides"].update({"train.learning_rate": .0002, "train.ent_coef": 0.0})
    run["initialize"].update(run=str(parent / "run"), checkpoint=str(source_checkpoint),
                              sha256=CHECKPOINT_SHA, restore_learner=False)
    for name, value in (("build-config.json", build), ("config.json", run)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    return build, run


def command(module, *args, name, seconds=600, train=False):
    env = dict(os.environ, METTA_SPATIAL_MUON_DENSE_ORIENTATION="canonical",
               METTA_SPATIAL_MUON_CONTEXT_MATRIX="1", METTA_SPATIAL_OPTIMIZER_LAYOUT="logical")
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
                        subprocess.run(["nvidia-smi", "--id=" + allocated_gpu(),
                                        "--query-gpu=timestamp,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader"], stdout=samples, stderr=log,
                                       timeout=3, check=True)
                    sampled = time.monotonic()
                console = OUT / "run/console.log"
                if console.exists():
                    text = console.read_text(errors="replace")
                    if "pooling:" in text and "SPATIAL_ADAPTER_ACTIVE module=integrations.direct_spatial_optimization" not in text:
                        process.terminate()
                        raise RuntimeError("Native worker entered Fabric setup without the verified spatial adapter")
                    if "NonFiniteGradsError" in text or "FloatingPointError" in text:
                        process.terminate()
                        raise RuntimeError("Nonfinite training result")
                    from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps
                    times = completed_epoch_times(text)
                    sps = interval_sps(times, span=2, steps_per_epoch=8192 * 256)
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
    subprocess.run(["nvidia-smi", "--id=" + allocated_gpu(), "--query-gpu=uuid,name,driver_version,memory.total,memory.used",
                    "--format=csv"], check=True, timeout=15)
    manifest = json.loads(Path("/work/input/source-manifest.json").read_text())
    for relative, digest in manifest.items():
        if hashlib.sha256((Path("/work/input") / relative).read_bytes()).hexdigest() != digest:
            raise ValueError("Input source or checkpoint hash differs: " + relative)
    prepare_configs()
    print("SMOKE_OK", jax.__version__, flush=True)


def build():
    command("launch_spatial_selfplay_training", "build", "--config", OUT / "build-config.json",
            "--output", OUT / "build", name="build-native", seconds=540)
    manifest = json.loads((OUT / "build/build.json").read_text())
    if manifest["model_sha256"] != "811e8de55c3fe327a670a362b076f88fc8e56b9525a347ed89c945cfe9e8863b":
        raise ValueError("Native model fingerprint changed during image migration")


def sampling_gate():
    for label in ("baseline", "candidate"):
        extra = ["--early-route-temperature", ".10", "--early-route-turns", "100"] if label == "candidate" else []
        command("evaluate_spatial_frozen_match", "--bundle", OUT / "self_bundle", "--run", PARENT / "run",
                "--opponent-bundle", OUT / "self_bundle", "--opponent-run", PARENT / "run",
                "--games", 512, "--pool-size", 512, "--seed", 35821, "--sample-seed", 31337,
                "--sampling-temperature", .05, "--split-sampling-temperature", .15,
                "--neutral-route-bias", 6, "--weak-owned-route-penalty", 4, "--doomed-attack-route-penalty", 4,
                *extra, "--output", OUT / "gate" / label, name="gate-" + label, seconds=360)
    command("analyze_spatial_frozen_match_pair", "--baseline", OUT / "gate/baseline",
            "--candidate", OUT / "gate/candidate", "--allow-policy-mode-change", "--seed", 35623,
            "--output", OUT / "sampling-gate.json", name="gate-pair", seconds=60)
    report = json.loads((OUT / "sampling-gate.json").read_text())
    if report["score_delta"] < -.12 or report["candidate_wld"][0] < 128:
        raise ValueError("Parent policy fails the training sampler gate")


def train():
    (OUT / "timing").mkdir()
    command("launch_spatial_selfplay_training", "train", "--config", OUT / "config.json",
            "--build", OUT / "build", "--output", OUT / "run", name="train-native", seconds=840, train=True)
    if not (OUT / "run/completed.json").is_file():
        raise ValueError("Trainer did not finish and validate its checkpoints")
    text = (OUT / "run/console.log").read_text()
    if f"DEVICE_ACTION_MASK_AUDIT actions={STEPS} illegal=0" not in text:
        raise ValueError("Complete legal-action audit missing")
    audits = [json.loads(line.split("DEVICE_REWARD_AUDIT ", 1)[1]) for line in text.splitlines()
              if line.startswith("DEVICE_REWARD_AUDIT ")]
    if not audits or audits[-1]["agent_steps"] != STEPS or audits[-1]["nonfinite_rewards"]:
        raise ValueError("Reward audit missing or nonfinite")
    reward_contract = json.loads((OUT / "reward-contract.json").read_text())
    if reward_contract["expected_unclipped"] and audits[-1]["native_clipped_rewards"]:
        raise ValueError("Scaled reward violates the audited native clamp bound")
    population = json.loads((OUT / "run/environments/8842/spatial-opponent-population.json").read_text())
    if (len(population["frozen_action_selection"]) != 10
            or population["opponent_weights"] != [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 6]
            or not all(c["0"] > 0 and c["0"] == c["1"] for c in population["counts"].values())):
        raise ValueError("Opponent pool or balanced seat allocation differs")
    from integrations.monitor_coworld_steady_interval import completed_epoch_times, interval_sps
    times = completed_epoch_times(text)
    sps = interval_sps(times, span=2, steps_per_epoch=8192 * 256)
    if sps is None or sps < 30_000:
        raise ValueError("Measured end-to-end training interval failed the SPS gate")
    (OUT / "training-audit.json").write_text(json.dumps(dict(
        environment_steps=STEPS, parallel_games=8192, horizon=256, minibatch=8192,
        replay_ratio=.5, steady_sps=sps, epoch_uptime=times, reward_audit=audits[-1]), indent=2) + "\n")


def export_and_audit(steps, bundle, suffix=""):
    checkpoint = OUT / f"run/checkpoints/metta_generals/run/{steps:016d}.bin"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    command("export_spatial_policy_bundle", "--build", OUT / "build/build.json", "--training", OUT / "run/training.json",
            "--checkpoint", checkpoint, "--sha256", digest, "--factory-source", SOURCE / "integrations/generals_fabric.py",
            "--output", bundle, "--serving-move-temperature", .05, "--serving-split-temperature", .15,
            "--serving-early-route-temperature", .1, "--serving-early-route-turns", 100,
            "--serving-neutral-route-bias", 6, "--serving-weak-owned-route-penalty", 4,
            "--serving-doomed-attack-route-penalty", 4, name="export" + suffix, seconds=180)
    report_path = OUT / ("checkpoint-serving-parity" + suffix + ".json")
    command("audit_spatial_checkpoint_serving_parity", "--bundle", bundle,
            "--replay-root", "/work/input/leader-root", "--factory-source", SOURCE / "integrations/generals_fabric.py",
            "--output", report_path, name="serving-parity" + suffix, seconds=300)
    report = json.loads(report_path.read_text())
    if (report["checkpoint_sha256"] != digest or report["public_states"] < 40
            or report["matching_top_actions"] != report["public_states"]
            or report["max_action_probability_difference"] > 1e-5
            or report["max_logit_difference"] > 2e-5 or report["max_rollout_transform_difference"] > 1e-5):
        raise ValueError("Trained checkpoint failed serving parity")


def evaluate():
    export_and_audit(STEPS, OUT / "bundle")
    arms = [("parent", OUT / "self_bundle"), ("child", OUT / "bundle")]
    if STEPS == 33_554_432:
        export_and_audit(16_777_216, OUT / "bundle-mid", "-mid")
        arms.append(("mid", OUT / "bundle-mid"))
    for label, bundle in arms:
        command("evaluate_spatial_population", "--bundle", bundle, "--population-build", OUT / "build/build.json",
                "--games", 4096, "--pool-size", 4096, "--seed", 37813, "--sample-seed", 8713,
                "--output", OUT / ("heldout-" + label), name="heldout-" + label, seconds=360)
    for label, _ in arms[1:]:
        command("analyze_spatial_population_pair", "--baseline", OUT / "heldout-parent",
                "--candidate", OUT / ("heldout-" + label), "--output", OUT / ("paired-" + label + ".json"),
                name="paired-" + label, seconds=60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "build", "sampling_gate", "train", "evaluate"))
    args = parser.parse_args()
    from integrations.cuda_runtime_binding import configure, audit
    configure()
    (OUT / ("cuda-binding-" + args.phase + ".json")).write_text(json.dumps(audit(), indent=2) + "\n")
    os.environ.update(PYTHONPATH=f"{SOURCE}/integrations/puffer_bootstrap:{SOURCE}:/opt/generals-source", JAX_PLATFORMS="cuda,cpu",
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", XDG_CACHE_HOME="/work/cache",
                      FABRIC_VERIFY_CACHE="/work/fabric-verify", JAX_COMPILATION_CACHE_DIR="/work/jax-cache",
                      TMPDIR="/work/tmp", METTA_PUFFER_SOURCE_REPOSITORY="/work/input/puffer.git",
                      METTA_PUFFER_RAYLIB_DIRECTORY="/work/input/raylib-5.5_linux_amd64")
    Path("/work/tmp").mkdir(exist_ok=True)
    globals()[args.phase]()


if __name__ == "__main__":
    main()

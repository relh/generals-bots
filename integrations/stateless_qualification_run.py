"""One bounded, uninstrumented stateless qualification; no continuation."""

import argparse
import json
from pathlib import Path

from integrations.row_rotation_trial import Trial, digest, read, write

CONFIG = Path(__file__).with_name("stateless_qualification")
PLAN = read(CONFIG / "plan.json")


def prepare(inputs, output):
    from integrations.classic_contract import validate_training_contract
    from integrations.native_spatial_asset import load_asset, training_contract

    if output.exists():
        raise ValueError("Qualification requires a fresh output directory")
    for name, expected in read(inputs / "source-manifest.json").items():
        if digest(inputs / name) != expected:
            raise ValueError("Sealed input differs: " + name)
    for item in PLAN["assets"]:
        directory = inputs / Path(item["destination"]).relative_to("/work/input")
        asset = load_asset(directory / "asset.json", manifest_sha256=item["manifest_sha256"])
        if asset.metadata["policy_sha256"] != item["policy_sha256"]:
            raise ValueError("Migrated policy identity differs")
    build, config = read(inputs / "build-config.json"), read(inputs / "config.json")
    for name, value in [("build-config.json", build), ("training-config.json", config)]:
        if digest(CONFIG / name) != PLAN["config_sha256"][name] or value != read(CONFIG / name):
            raise ValueError("Qualification config differs: " + name)
    asset = load_asset(inputs / "assets/cold/asset.json", manifest_sha256=config["initialize"]["manifest_sha256"])
    if config["initialize"]["restore_learner"] or config["total_timesteps"] != 4_194_304:
        raise ValueError("Qualification requires fresh optimizer and exactly eight epochs")
    if (
        training_contract(build["python_environment"]["options"], config["overrides"])
        != asset.metadata["training_contract"]
    ):
        raise ValueError("Source objective changed")
    output.mkdir(parents=True)
    write(output / "plan.json", PLAN)
    write(output / "candidate/classic-contract.json", validate_training_contract(build, config))
    write(output / "candidate/build-config.json", build)
    write(output / "candidate/qualification/config.json", config)
    write(
        output / "source-binding.json",
        dict(
            input_manifest_sha256=digest(inputs / "source-manifest.json"),
            source_asset_sha256=digest(inputs / "assets/cold/asset.json"),
            source_policy_sha256=asset.metadata["policy_sha256"],
            parameter_count=asset.metadata["parameter_count"],
        ),
    )


def audit(output):
    from integrations.monitor_coworld_steady_interval import completed_epoch_times

    stage = output / "candidate/qualification"
    report = read(stage / "training-audit.json")
    times = completed_epoch_times((stage / "run/console.log").read_text())
    if set(times) != set(range(1, 9)):
        raise ValueError("Require exactly eight completed epochs")
    elapsed = times[8] - times[2]
    if elapsed <= 0 or 3_145_728 / elapsed < 30_000:
        raise ValueError("Full post-warmup interval failed 30K SPS")
    for path in (output / "source-serving-parity.json", stage / "serving-parity.json"):
        parity = read(path)
        if (
            parity["public_states"] != 46
            or parity["matching_top_actions"] != 46
            or parity["inference_backend"] != "gpu"
        ):
            raise ValueError("Require 46 GPU native-serving parity fixtures")
    hardware = [json.loads(line) for line in (stage / "hardware.jsonl").read_text().splitlines()]
    if not hardware or any("H100" not in row["gpu"]["name"] for row in hardware):
        raise ValueError("Require retained H100 hardware telemetry")
    uuids = {row["gpu"]["uuid"] for row in hardware}
    if len(uuids) != 1:
        raise ValueError("GPU identity changed")
    checkpoint = stage / "run/checkpoints/metta_generals/run/0000000004194304.bin"
    from integrations.learner_checkpoint import LearnerCheckpoint

    learner_path = checkpoint.with_name(checkpoint.name + ".learner")
    binding = read(output / "source-binding.json")
    initial = stage / "run/initial-policy.bin"
    if digest(initial) != PLAN["initialization"]["weights"] or initial.with_name(initial.name + ".learner").exists():
        raise ValueError("Fresh optimizer or exact source weight initialization differs")
    if checkpoint.stat().st_size != 4 * binding["parameter_count"]:
        raise ValueError("Checkpoint parameter count differs from selected source")
    learner = LearnerCheckpoint.from_bytes(learner_path.read_bytes(), binding["parameter_count"])
    if learner.agent_steps != 4_194_304 or learner.epoch != 8:
        raise ValueError("Final learner clock differs")
    write(
        output / "qualified.json",
        dict(
            qualified=True,
            continuation_authorized=False,
            measured_steps=3_145_728,
            interval_seconds=elapsed,
            steady_sps=3_145_728 / elapsed,
            sampled_peak_gpu_memory_mib=max(float(row["gpu"]["memory.used"].split()[0]) for row in hardware),
            telemetry_cadence_seconds=5,
            telemetry_scope="whole training subprocess; samples are not aligned to the steady epoch interval",
            first_observed_memory_mib=float(hardware[0]["gpu"]["memory.used"].split()[0]),
            first_observed_time_unix=hardware[0]["time_unix"],
            whole_process_mean_gpu_utilization_percent=sum(
                float(row["gpu"]["utilization.gpu"].split()[0]) for row in hardware
            )
            / len(hardware),
            gpu_uuid=next(iter(uuids)),
            training=report,
            evidence={
                str(path.relative_to(output)): digest(path)
                for path in (
                    initial,
                    checkpoint,
                    learner_path,
                    stage / "rotation-audit.json",
                    stage / "training-audit.json",
                    stage / "serving-parity.json",
                    output / "source-serving-parity.json",
                    stage / "hardware.jsonl",
                )
            },
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "build", "qualify"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.input, args.output)
        return
    from integrations.cuda_runtime_binding import configure

    configure()
    trial = Trial(args.input, args.output)
    if args.phase == "qualify":
        trial.train("candidate", "qualification")
        audit(args.output)
    else:
        getattr(trial, args.phase)()
        if args.phase == "build":
            build = args.output / "candidate/build"
            if read(build / "build.json")["model_state_words"] != 0:
                raise ValueError("Native build retained external carry")
            if "bridge_profile" in (build / "source/src/pufferl.cu").read_text():
                raise ValueError("Qualification must be uninstrumented")


if __name__ == "__main__":
    main()

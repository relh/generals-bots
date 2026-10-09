"""Final evidence checks for qualification and restored training stages."""
import json
from integrations.policy_trial import digest, read, write
from integrations.training_inputs import ASSETS

def qualification_audit(output, *, destination):
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
    if digest(initial) != ASSETS[0]["policy_sha256"] or initial.with_name(initial.name + ".learner").exists():
        raise ValueError("Fresh optimizer or exact source weight initialization differs")
    if checkpoint.stat().st_size != 4 * binding["parameter_count"]:
        raise ValueError("Checkpoint parameter count differs from selected source")
    learner = LearnerCheckpoint.from_bytes(learner_path.read_bytes(), binding["parameter_count"])
    if learner.agent_steps != 4_194_304 or learner.epoch != 8:
        raise ValueError("Final learner clock differs")
    write(
        destination,
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
                    stage / "run/native-admission.json",
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

def clock(path, words, steps, epoch):
    from integrations.learner_checkpoint import LearnerCheckpoint
    state = LearnerCheckpoint.from_bytes(path.read_bytes(), words)
    if state.agent_steps != steps or state.epoch != epoch:
        raise ValueError('Authenticated learner cumulative clock differs')

def finish(output):
    stage = output / 'candidate/continuation'
    asset = read(stage / 'asset/asset.json')
    clock(stage / 'asset/policy.bin.learner', asset['parameter_count'], 33554432, 64)
    audit = read(stage / 'training-audit.json')
    if (audit['starting_agent_steps'] != 4194304 or audit['ending_agent_steps'] != 33554432
            or audit['environment_steps'] != 29360128 or audit['steady_sps'] < 30000):
        raise ValueError('Continuation clock or strict throughput gate differs')

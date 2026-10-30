#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-sentinel-smoke
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/sentinel-smoke-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
train_job_id=${TRAIN_JOB_ID:?training Slurm job ID required}
heldout="$workspace/gpu-batch-four-long-${train_job_id}-classic-heldout-summary.json"
test -s "$heldout"
test -s "$workspace/source-sentinel-classic10/integrations/configs/gpu-batch-sentinel-transfer-build.json"

if python3 - "$heldout" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
assert summary["seeds"] == [1001, 1002, 1003, 1004, 1005]
assert len(summary["policies"]) == 1
policy = next(iter(summary["policies"].values()))
assert policy["games"] >= 80
print(f"Current classic held-out performance: {policy['perf']:.6f}", flush=True)
sys.exit(0 if policy["perf"] >= 0.60 else 3)
PY
then
    echo "Held-out target passed; skipping Sentinel smoke."
    exit 0
else
    status=$?
    [[ "$status" == 3 ]] || exit "$status"
fi

export TRAIN_JOB_ID="$train_job_id"
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=64 --cpu-bind=cores bash -s <<'RUN'
set -euo pipefail
workspace=/tmp/relh-generals-gpu
build_name="build-sentinel-transfer-${SLURM_JOB_ID}"
test ! -e "$workspace/$build_name"

docker run --rm --name "relh-generals-sentinel-build-${SLURM_JOB_ID}" \
    --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
    --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-sentinel-classic10 \
    relh-generals-b300:20260923b \
    bash /work/source-sentinel-classic10/integrations/run_puffer_gpu.sh build \
    --config /work/source-sentinel-classic10/integrations/configs/gpu-batch-sentinel-transfer-build.json \
    --output "/work/$build_name"

python3 - "$workspace" "$TRAIN_JOB_ID" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
train_job_id, job_id = sys.argv[2:]
summary = json.loads((workspace / f"gpu-batch-four-long-{train_job_id}-classic-heldout-summary.json").read_text())
selected, policy = next(iter(summary["policies"].items()))
run = workspace / f"gpu-batch-four-long-{train_job_id}-{selected}"
checkpoint = run / policy["checkpoint_relative_path"]
assert checkpoint.is_relative_to(run / "checkpoints")
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == policy["checkpoint_sha256"]
source_build = json.loads((run / "run.json").read_text())["build"]
target_build = json.loads((workspace / f"build-sentinel-transfer-{job_id}" / "build.json").read_text())
assert source_build["model_sha256"] == target_build["model_sha256"]
assert source_build["model_state_words"] == target_build["model_state_words"]
assert source_build["revision"] == target_build["revision"]
base = json.loads((workspace / "gpu-batch-four-long-base.json").read_text())
for index in range(4):
    config = json.loads(json.dumps(base))
    config["total_timesteps"] = 65_536
    config["seed"] = 341 + index
    config["initialize"] = {
        "run": str(run).replace(str(workspace), "/work"),
        "checkpoint": str(checkpoint).replace(str(workspace), "/work"),
        "sha256": policy["checkpoint_sha256"],
        "allow_environment_transfer": True,
    }
    (workspace / f"sentinel-smoke-{job_id}-{index}.json").write_text(json.dumps(config))
print(f"Sentinel smoke initialized from policy {selected} at {checkpoint}", flush=True)
PY

(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits || true
        sleep 5
    done
) > "$workspace/sentinel-smoke-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
sampler=$!
cleanup() {
    kill "$sampler" 2>/dev/null || true
    for index in 0 1 2 3; do
        docker stop -t 5 "relh-generals-sentinel-smoke-${SLURM_JOB_ID}-${index}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
pids=()
for index in 0 1 2 3; do
    (
        timeout -k 30s 2h docker run --rm \
            --name "relh-generals-sentinel-smoke-${SLURM_JOB_ID}-${index}" \
            --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
            --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
            -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-sentinel-classic10 \
            relh-generals-b300:20260923b \
            bash /work/source-sentinel-classic10/integrations/run_puffer_gpu.sh train \
            --config "/work/sentinel-smoke-${SLURM_JOB_ID}-${index}.json" \
            --build "/work/$build_name" \
            --output "/work/sentinel-smoke-${SLURM_JOB_ID}-${index}" \
            > "$workspace/sentinel-smoke-${SLURM_JOB_ID}-${index}.log" 2>&1
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

python3 - "$workspace" "$SLURM_JOB_ID" <<'PY'
import json
import statistics
import sys
from pathlib import Path

workspace, job_id = Path(sys.argv[1]), sys.argv[2]
samples = []
for line in (workspace / f"sentinel-smoke-gpu-{job_id}.csv").read_text().splitlines():
    fields = line.split(",")
    if len(fields) >= 3:
        try:
            samples.append(int(fields[0].strip()))
        except ValueError:
            pass
for index in range(4):
    run = workspace / f"sentinel-smoke-{job_id}-{index}"
    completed = json.loads((run / "completed.json").read_text())
    assert completed["trained_timesteps"] == 65_536
active = samples[5:-2]
utilization = statistics.median(active) if active else 0
print(f"Sentinel smoke: four 65,536-step policies, {len(samples)} GPU samples, "
      f"median active GPU utilization {utilization:.1f}%", flush=True)
if len(active) < 5 or utilization < 50:
    raise SystemExit("Sentinel smoke did not establish adequate GPU utilization")
PY
RUN

#!/usr/bin/env bash
# Submit only after the bounded Sentinel smoke has established GPU utilization.
# Example: sbatch --dependency=afterok:<smoke-job-id> \
#   --export=ALL,TRAIN_JOB_ID=<first-training-job-id>,SMOKE_JOB_ID=<smoke-job-id> \
#   integrations/generals_b300_sentinel_long.sbatch
#SBATCH --job-name=relh-generals-sentinel-long
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=27:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/sentinel-long-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
train_job_id=${TRAIN_JOB_ID:?first training Slurm job ID required}
smoke_job_id=${SMOKE_JOB_ID:?Sentinel smoke Slurm job ID required}
[[ "$train_job_id" =~ ^[0-9]+$ && "$smoke_job_id" =~ ^[0-9]+$ ]]
test -s "$workspace/gpu-batch-four-long-${train_job_id}-classic-proof.json"

# The independent verifier returns 3 only when all evidence is valid and the
# held-out score is below target. Any malformed evidence stops the job.
if python3 "$workspace/verify_classic_heldout.py" "$workspace" "$train_job_id"; then
    echo "The first policy passed the held-out target; skipping Sentinel training."
    exit 0
else
    status=$?
    [[ "$status" == 3 ]] || exit "$status"
fi

test -s "$workspace/sentinel-smoke-gpu-${smoke_job_id}.csv"
test -s "$workspace/build-sentinel-transfer-${smoke_job_id}/build.json"
test -s "$workspace/gpu-batch-four-long-base.json"
test -f "$workspace/source-sentinel-classic10/integrations/run_puffer_gpu.sh"

python3 - "$workspace" "$smoke_job_id" <<'PY'
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
active = samples[5:-2]
assert len(active) >= 5 and statistics.median(active) >= 50, "Sentinel smoke underused the GPU"
for index in range(4):
    run = workspace / f"sentinel-smoke-{job_id}-{index}"
    completed = json.loads((run / "completed.json").read_text())
    assert completed["trained_timesteps"] == 65_536, f"smoke policy {index} incomplete"
print(f"Sentinel smoke median active GPU utilization: {statistics.median(active):.1f}%", flush=True)
PY

export TRAIN_JOB_ID="$train_job_id" SMOKE_JOB_ID="$smoke_job_id"
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=64 --cpu-bind=cores bash -s <<'RUN'
set -euo pipefail
workspace=/tmp/relh-generals-gpu
run_prefix="sentinel-long-${SLURM_JOB_ID}"

python3 - "$workspace" "$TRAIN_JOB_ID" "$SMOKE_JOB_ID" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
train_job_id, smoke_job_id, job_id = sys.argv[2:]
summary = json.loads((workspace / f"gpu-batch-four-long-{train_job_id}-classic-heldout-summary.json").read_text())
selected, policy = next(iter(summary["policies"].items()))
run = workspace / f"gpu-batch-four-long-{train_job_id}-{selected}"
checkpoint = run / policy["checkpoint_relative_path"]
assert checkpoint.is_relative_to(run / "checkpoints")
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == policy["checkpoint_sha256"]
source_build = json.loads((run / "run.json").read_text())["build"]
target_build = json.loads((workspace / f"build-sentinel-transfer-{smoke_job_id}" / "build.json").read_text())
for field in ("model_sha256", "model_state_words", "revision"):
    assert source_build[field] == target_build[field], f"incompatible model {field}"
base = json.loads((workspace / "gpu-batch-four-long-base.json").read_text())
for index in range(4):
    config = json.loads(json.dumps(base))
    config["total_timesteps"] = 31_457_280
    config["seed"] = 441 + index
    config["initialize"] = {
        "run": str(run).replace(str(workspace), "/work"),
        "checkpoint": str(checkpoint).replace(str(workspace), "/work"),
        "sha256": policy["checkpoint_sha256"],
        "allow_environment_transfer": True,
    }
    (workspace / f"sentinel-long-{job_id}-{index}.json").write_text(json.dumps(config))
print(f"Starting Sentinel transfer from policy {selected} at {checkpoint}", flush=True)
PY

(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits || true
        sleep 10
    done
) > "$workspace/${run_prefix}-gpu.csv" 2>&1 &
sampler=$!
cleanup() {
    kill "$sampler" 2>/dev/null || true
    for index in 0 1 2 3; do
        docker stop -t 5 "relh-generals-sentinel-long-${SLURM_JOB_ID}-${index}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
pids=()
for index in 0 1 2 3; do
    (
        timeout -k 30s 26h docker run --rm \
            --name "relh-generals-sentinel-long-${SLURM_JOB_ID}-${index}" \
            --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
            --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
            -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-sentinel-classic10 \
            relh-generals-b300:20260923b \
            bash /work/source-sentinel-classic10/integrations/run_puffer_gpu.sh train \
            --config "/work/sentinel-long-${SLURM_JOB_ID}-${index}.json" \
            --build "/work/build-sentinel-transfer-${SMOKE_JOB_ID}" \
            --output "/work/sentinel-long-${SLURM_JOB_ID}-${index}" \
            > "$workspace/sentinel-long-${SLURM_JOB_ID}-${index}.log" 2>&1
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
for index in 0 1 2 3; do
    python3 - "$workspace/sentinel-long-${SLURM_JOB_ID}-${index}/completed.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as result:
    assert json.load(result)["trained_timesteps"] == 31_457_280
PY
done
# Keep the allocation alive for the mettabox to archive final checkpoints.
sleep 150
RUN

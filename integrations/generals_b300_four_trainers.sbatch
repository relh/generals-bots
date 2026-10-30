#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-four-trainers
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/four-trainers-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
test -f "$workspace/build-gpu-wide16x16-6021/build.json"
test -f "$workspace/gpu-batch-four-long-base.json"

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=64 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
base = json.loads((workspace / "gpu-batch-four-long-base.json").read_text())
for index in range(4):
    config = json.loads(json.dumps(base))
    config["seed"] = 241 + index
    (workspace / f"gpu-batch-four-long-{index}.json").write_text(json.dumps(config))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits || true
            sleep 10
        done
    ) > "$workspace/four-long-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        for index in 0 1 2 3; do
            docker stop -t 5 "relh-generals-four-long-${SLURM_JOB_ID}-${index}" >/dev/null 2>&1 || true
        done
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    trap "exit 130" INT
    pids=()
    for index in 0 1 2 3; do
        (
            timeout -k 30s 15h docker run --rm --name "relh-generals-four-long-${SLURM_JOB_ID}-${index}" \
                --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
                --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
                -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
                relh-generals-b300:20260923b \
                bash /work/source-gpu/integrations/run_puffer_gpu.sh train \
                --config "/work/gpu-batch-four-long-${index}.json" \
                --build /work/build-gpu-wide16x16-6021 \
                --output "/work/gpu-batch-four-long-${SLURM_JOB_ID}-${index}" \
                > "$workspace/four-long-${SLURM_JOB_ID}-${index}.log" 2>&1
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for index in 0 1 2 3; do
        python3 - "$workspace/gpu-batch-four-long-${SLURM_JOB_ID}-${index}/completed.json" <<"PY"
import json
import sys

with open(sys.argv[1]) as result:
    assert json.load(result)["trained_timesteps"] == 31_457_280
PY
    done
    # Let the mettabox archiver copy and verify final checkpoints before this allocation exits.
    sleep 150
'

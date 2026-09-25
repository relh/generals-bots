#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-throughput-smoke
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/throughput-smoke-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
build = json.loads((workspace / "generals-wide16-build.json").read_text())
build["python_environment"]["options"]["parallel_games"] = 1024
build["python_environment"]["spec"]["agents"] = 1024
(workspace / "throughput-1024-build.json").write_text(json.dumps(build))
train = json.loads((workspace / "gpu-batch-pilot.json").read_text())
train["total_timesteps"] = 1048576
train["overrides"].update({"vec.total_agents": 1024, "vec.num_threads": 1, "train.minibatch_size": 1024})
(workspace / "throughput-1024-train.json").write_text(json.dumps(train))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 5
        done
    ) > "$workspace/throughput-smoke-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "relh-generals-throughput-${SLURM_JOB_ID}" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 16m docker run --rm --name "relh-generals-throughput-${SLURM_JOB_ID}" \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-gpu/integrations/run_puffer_gpu.sh build \\
                --config /work/throughput-1024-build.json \\
                --output /work/build-throughput-1024-${SLURM_JOB_ID}
            bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                --config /work/throughput-1024-train.json \\
                --build /work/build-throughput-1024-${SLURM_JOB_ID} \\
                --output /work/throughput-1024-${SLURM_JOB_ID}
        "
'

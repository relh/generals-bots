#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-two-goalaction-trainers-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:16:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/two-goalaction-trainers-probe-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    test -s "$workspace/build-throughput-goalaction8-classic4m-8731/build.json"
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
source = json.loads((workspace / "throughput-goalaction8-classic4m-train.json").read_text())
for index in range(2):
    config = json.loads(json.dumps(source))
    config["seed"] = 391 + index
    config["total_timesteps"] = 4_194_304
    config["overrides"]["base.checkpoint_interval"] = 8
    (workspace / f"two-goalaction-trainers-probe-{index}.json").write_text(json.dumps(config))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/two-goalaction-trainers-probe-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-two-goalaction-trainers-probe-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 14m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=24 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            pids=()
            for index in 0 1; do
                bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                    --config /work/two-goalaction-trainers-probe-\${index}.json \\
                    --build /work/build-throughput-goalaction8-classic4m-8731 \\
                    --output /work/two-goalaction-trainers-probe-${SLURM_JOB_ID}-\${index} \\
                    > /work/two-goalaction-trainers-probe-${SLURM_JOB_ID}-\${index}.launch.log 2>&1 &
                pids+=(\$!)
            done
            for pid in \${pids[@]}; do wait \$pid; done
        "
'

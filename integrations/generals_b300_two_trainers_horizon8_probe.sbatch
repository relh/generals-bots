#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-two-trainers-horizon8-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:09:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/two-trainers-horizon8-probe-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    test -s "$workspace/build-throughput-jitmix4m-7556/build.json"
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
source = json.loads((workspace / "throughput-jitmix4m-train.json").read_text())
for index in range(2):
    config = json.loads(json.dumps(source))
    config["seed"] = 261 + index
    config["total_timesteps"] = 4_194_304
    config["overrides"]["train.horizon"] = 8
    config["overrides"]["base.checkpoint_interval"] = 32
    (workspace / f"two-trainers-horizon8-probe-{index}.json").write_text(json.dumps(config))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/two-trainers-horizon8-probe-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-two-trainers-horizon8-probe-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 7m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=24 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            pids=()
            for index in 0 1; do
                bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                    --config /work/two-trainers-horizon8-probe-\${index}.json \\
                    --build /work/build-throughput-jitmix4m-7556 \\
                    --output /work/two-trainers-horizon8-probe-${SLURM_JOB_ID}-\${index} \\
                    > /work/two-trainers-horizon8-probe-${SLURM_JOB_ID}-\${index}.launch.log 2>&1 &
                pids+=(\$!)
            done
            for pid in \${pids[@]}; do wait \$pid; done
        "
'

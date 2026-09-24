#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-mlp-dagger-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/mlp-dagger-pilot-%j.log
set -euo pipefail

export TRAIN_JOB_ID=${TRAIN_JOB_ID:?training Slurm job ID required}
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    run="$workspace/throughput-mlp-dagger4m-${TRAIN_JOB_ID}"
    test -s "$run/completed.json"
    python3 - <<"PY"
import json
import os
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
job = os.environ["SLURM_JOB_ID"]
build = json.loads((workspace / "throughput-mlp-dagger4m-build.json").read_text())
options = build["python_environment"]["options"]
options["horizon"] = 800
options["classic_maps"] = True
assert options["board_size"] == 10 and options["opponent"] == "mixed"
assert options["parallel_games"] == build["python_environment"]["spec"]["agents"] == 1024
(workspace / f"mlp-classic-build-{job}.json").write_text(json.dumps(build))
evaluation = {
    "seeds": [901, 902],
    "episodes_per_seed": 1,
    "timeout_seconds": 900,
    "allow_environment_transfer": True,
}
(workspace / f"mlp-dagger-pilot-{job}.json").write_text(json.dumps(evaluation))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/mlp-dagger-pilot-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-mlp-dagger-pilot-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 10m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-gpu/integrations/run_puffer_gpu.sh build \\
                --config /work/mlp-classic-build-${SLURM_JOB_ID}.json \\
                --output /work/build-mlp-dagger-${SLURM_JOB_ID}
            bash /work/source-gpu/integrations/eval_puffer_gpu.sh \\
                /work/throughput-mlp-dagger4m-${TRAIN_JOB_ID} \\
                /work/build-mlp-dagger-${SLURM_JOB_ID} \\
                /work/mlp-dagger-pilot-${SLURM_JOB_ID} \\
                /work/mlp-dagger-pilot-${SLURM_JOB_ID}.json
        "
'

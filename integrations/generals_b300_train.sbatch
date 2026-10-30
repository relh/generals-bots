#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-gpu-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/pilot-v3-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
test -f "$workspace/build-gpu-batch-v5/build.json"
test -f "$workspace/gpu-batch-pilot.json"

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    docker run --rm --gpus "device=${SLURM_STEP_GPUS}" \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -w /work \
        -e HOME=/work \
        -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash /work/source-gpu/integrations/run_puffer_gpu.sh train \
        --config /work/gpu-batch-pilot.json \
        --build /work/build-gpu-batch-v5 \
        --output /work/gpu-batch-pilot-v3
'

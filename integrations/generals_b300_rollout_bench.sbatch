#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-rollout-bench
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/rollout-bench-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
test -f "$workspace/source-gpu/integrations/bench_puffer_rollout.py"

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    args=(--native-encode --parallel-games 256 1024)
    timeout -k 10s 12m docker run --rm --name "relh-generals-rollout-bench-${SLURM_JOB_ID}" \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=8 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
        relh-generals-b300:20260923b \
        python -u /work/source-gpu/integrations/bench_puffer_rollout.py "${args[@]}"
'

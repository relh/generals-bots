#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-gpu-30m
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/long-%j.log
set -euo pipefail

# Mandatory user policy: verify lowest scheduling priority before workload startup.
slurm_task_priority_record=$(scontrol show job -o "${SLURM_JOB_ID:?}")
[[ " $slurm_task_priority_record " == *" Nice=2147483645 "* ]] || { printf 'Lowest Slurm Nice verification failed\n' >&2; exit 1; }
slurm_task_priority=unknown
for slurm_task_field in $slurm_task_priority_record; do
  case "$slurm_task_field" in Priority=*) slurm_task_priority=${slurm_task_field#Priority=} ;; esac
done
[[ "$slurm_task_priority" =~ ^[0-9]+$ ]] || exit 1
printf 'SLURM_PRIORITY_READBACK JobId=%s Nice=2147483645 Priority=%s\n' "$SLURM_JOB_ID" "$slurm_task_priority" >&2
unset slurm_task_priority_record slurm_task_field


workspace=/tmp/relh-generals-gpu
test -f "$workspace/build-gpu-batch-v5/build.json"
test -f "$workspace/gpu-batch-30m.json"

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    docker run --rm --gpus "device=${SLURM_STEP_GPUS}" \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -w /work \
        -e HOME=/work \
        -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash /work/source-gpu/integrations/run_puffer_gpu.sh train \
        --config /work/gpu-batch-30m.json \
        --build /work/build-gpu-batch-v5 \
        --output /work/gpu-batch-30m
'

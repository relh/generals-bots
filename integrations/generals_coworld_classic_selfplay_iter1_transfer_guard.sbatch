#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-transfer-guard
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-transfer-guard-%j.log
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

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=2 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
w=/tmp/relh-generals-coworld
printf '%s  %s\n' e56cd5a8834af5231441404cd51366259d65a5009e0c000e7682df0eecda084f "$disk/puffer_selfplay_iter1_transfer-v1-20260927.py" | sha256sum -c -
printf '%s  %s\n' 2e7bfee68578ef7ac65608e343cb3bbe604ac38341bb095dae9f292323db80cb "$disk/audit_coworld_selfplay_iter1_transfer_guard-v1-20260927.py" | sha256sum -c -
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
timeout -k 10s 3m docker run --rm --gpus "device=$gpu" --cpus=2 --ulimit core=0:0 \
    --user "$(id -u):$(id -g)" -v "$disk:/recovery" -v "$w:/work" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b \
    python /recovery/audit_coworld_selfplay_iter1_transfer_guard-v1-20260927.py
NODE

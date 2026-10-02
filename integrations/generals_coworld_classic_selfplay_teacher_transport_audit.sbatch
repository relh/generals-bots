#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-teacher-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:07:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-teacher-audit-%j.log
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

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
adapter=$disk/metta_puffer-selfplay-teacher-v2-20260927.py
audit=$disk/audit_coworld_selfplay_teacher_transport-v1-20260927.py
printf '%s  %s\n' a0261c5d81806b8ba2b4ed9675dda0dc51bd08c29109c6ef92dc3cb26c470886 "$adapter" | sha256sum -c -
printf '%s  %s\n' befead5dcb5c877edcd41fba52e1323696d97dba887d9b80a91eff5ec9df7662 "$audit" | sha256sum -c -
test -f "$disk/classic-spatial4-hintcurriculum-prior05-32m-build-27745/build.json"
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
IFS=, read -r gpu memory utilization <<< "$usage"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-selfplay-teacher-audit-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
timeout -k 15s 6m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=4 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" \
    -v "$adapter:/work/source-spatial2/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery/classic-selfplay-teacher-audit-$SLURM_JOB_ID/cache \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python "/recovery/$(basename "$audit")" \
    --build /recovery/classic-spatial4-hintcurriculum-prior05-32m-build-27745/build.json --turns 32
NODE

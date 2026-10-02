#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-wide-transport
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:18:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/wide-transport-profile-%j.log
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


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
printf "%s  %s\n" 918ef1fcc3a409e1338090d52d4758b152fc91854e29a9d8429f2fb8dfda8b79 \
    "$w/metta_puffer-leanprofile-16245.py" | sha256sum -c -
printf "%s  %s\n" 4dc65235348161663fa47e05876d5ab20d33520dc5bcdfdc51943df4d8ecc442 \
    "$w/profile-coworld-lean-16245.py" | sha256sum -c -
printf "%s  %s\n" bd35a2359a62ee8160511f91f2244a92bb883a6ff5fa22802cb4d0a4ca0ca9c6 \
    "$w/metta_environment-main-fast-16245.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; break; fi
done
test -n "$gpu"
name=relh-classic-wide-transport-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for games in 4096 8192 16384; do
    echo "PROFILE games=$games"
    timeout -k 20s 5m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" \
        -v "$w/metta_puffer-leanprofile-16245.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
        -v "$w/metta_environment-main-fast-16245.py:/work/oldsource/metta_training/environment.py:ro" \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource \
        relh-generals-b300:20260923b python3 /work/profile-coworld-lean-16245.py \
        --build /work/normalized-clean-build-16788/build.json \
        --games "$games" --warmup 6 --steps 16 --transport
done
'

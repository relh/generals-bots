#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-transport-profile
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/transport-profile-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
source_dir=$w/source-normalized-16245
printf "%s  %s\n" 5aca1a34e5a35b0277bd909a72f3bd1293f134dde6d41f59c8ea56fb09bbcf9e \
    "$w/profile-coworld-transport-16245.py" | sha256sum -c -
printf "%s  %s\n" 813612948bee564284c159e318267de102fe668dc31048ab8f94833688da7b0a \
    "$w/metta_puffer-half-16245.py" | sha256sum -c -
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
name=relh-classic-transport-profile-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for half in 0 1; do
  for games in 1024 4096; do
    echo "PROFILE half=$half games=$games"
    extra=()
    if (( half )); then extra+=(--half-precision-observation); fi
    timeout -k 15s 3m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=4 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" \
        -v "$w/metta_puffer-half-16245.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource \
        relh-generals-b300:20260923b python3 \
        /work/profile-coworld-transport-16245.py \
        --build /work/small-to-classic-ppo-build-16245/build.json \
        --games "$games" --warmup 4 --steps 16 --transport "${extra[@]}"
  done
done
'

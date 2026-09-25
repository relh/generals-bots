#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-long
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/device-long-profile-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
printf "%s  %s\n" 9f29360f4d7062dac040082b6a7b280d7deea4ad2bf051c8b3a2d2a22907aac4 \
    "$w/profile_coworld_device_step.py" | sha256sum -c -
printf "%s  %s\n" 918ef1fcc3a409e1338090d52d4758b152fc91854e29a9d8429f2fb8dfda8b79 \
    "$w/metta_puffer-leanprofile-16245.py" | sha256sum -c -
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
name=relh-classic-device-long-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for games in 4096; do
    echo "PROFILE games=$games"
    timeout -k 20s 6m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" \
        -v "$w/metta_puffer-leanprofile-16245.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
        -v "$w/metta_environment-main-fast-16245.py:/work/oldsource/metta_training/environment.py:ro" \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource \
        relh-generals-b300:20260923b python3 /work/profile_coworld_device_step.py \
        --build /work/normalized-clean-build-16788/build.json \
        --games "$games" --warmup 16 --steps 1300
done
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-lean-transport
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
#SBATCH --output=/tmp/relh-generals-coworld/lean-transport-profile-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
printf "%s  %s\n" 918ef1fcc3a409e1338090d52d4758b152fc91854e29a9d8429f2fb8dfda8b79 \
    "$w/metta_puffer-leanprofile-16245.py" | sha256sum -c -
printf "%s  %s\n" 4dc65235348161663fa47e05876d5ab20d33520dc5bcdfdc51943df4d8ecc442 \
    "$w/profile-coworld-lean-16245.py" | sha256sum -c -
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
name=relh-classic-lean-transport-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for lean in 0 1; do
  for games in 1024 4096; do
    echo "PROFILE lean=$lean games=$games"
    extra=()
    if (( lean )); then extra+=(--lean); fi
    timeout -k 15s 3m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=4 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" \
        -v "$w/metta_puffer-leanprofile-16245.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource \
        relh-generals-b300:20260923b python3 /work/profile-coworld-lean-16245.py \
        --build /work/small-to-classic-ppo-build-16245/build.json \
        --games "$games" --warmup 4 --steps 16 --transport "${extra[@]}"
  done
done
'

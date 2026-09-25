#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-hybrid-clean
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/hybrid-clean-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-hybrid-clean-$job
output=$workspace/hybrid-clean-$job
mkdir -p "$output"
printf "%s  %s\n" ae3f68b80edcee43dd2968eb438e2dbb6d7f156f4fae732079989f0b7c8f86cb \
    "$workspace/source-hybrid-teacher/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e \
    "$workspace/hybrid-periodic-build-13829/build.json" | sha256sum -c -
printf "%s  %s\n" eaa4075cf329dcae53a05e5dcf54a258075a27c33be7d7b1a4b5389da7e4bda5 \
    "$workspace/hybrid-periodic-build-13829/puffer" | sha256sum -c -
printf "%s  %s\n" eead2d80b8e88ad437288faac9e011a6ca609b860ae8d78c24152f6654dc3202 \
    "$workspace/hybrid-periodic-13829/run/checkpoints/metta_generals/run/0000000004194304.bin" | sha256sum -c -
python3 - "$workspace/hybrid-periodic-13829/config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
assert config["total_timesteps"] == 4_194_304
assert config["overrides"]["vec.total_agents"] == 4096
assert config["overrides"]["vec.num_buffers"] == 4
config["total_timesteps"] = 33_554_432
config["seed"] = 625
config["initialize"] = {
    "run": "/work/hybrid-periodic-13829/run",
    "checkpoint": "/work/hybrid-periodic-13829/run/checkpoints/metta_generals/run/0000000004194304.bin",
    "sha256": "eead2d80b8e88ad437288faac9e011a6ca609b860ae8d78c24152f6654dc3202",
}
output.write_text(json.dumps(config) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if (( ${#allocated_gpus[@]} != ${#usage[@]} )); then
    echo "Slurm allocation and visible GPU list differ" >&2
    exit 1
fi
chosen_gpu=
gpu_local=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then
        chosen_gpu=${allocated_gpus[$i]}
        gpu_local=$i
        break
    fi
done
if [[ -z $chosen_gpu ]]; then
    echo "No uncontended B300 in Slurm allocation $SLURM_STEP_GPUS" >&2
    exit 1
fi
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$chosen_gpu local_index=$gpu_local"
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-hybrid-teacher:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
ln -s "$output/run" "$workspace/hybrid-clean-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/hybrid-clean-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 40m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-hybrid-teacher/integrations/run_puffer_gpu.sh train \
    --config "/work/hybrid-clean-$job/config.json" \
    --build /work/hybrid-periodic-build-13829 \
    --output "/work/hybrid-clean-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-hybrid-teacher/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" hybrid-clean 1; then
        echo "Hybrid long run stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

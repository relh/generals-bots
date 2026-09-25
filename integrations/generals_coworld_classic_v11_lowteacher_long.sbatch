#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-v11-lowteacher-long
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:28:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/v11-lowteacher-long-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/v11-lowteacher-long-$job
name=relh-classic-v11-lowteacher-long-$job
mkdir -p "$out"
printf "%s  %s\n" 459e10eab59671c160d11ae8f49a3ffa31e487adf3a606ac7fedc3748745a56d "$w/source-v11-strongmix/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 48a47da8c0ade533e0b61102462911e54dc456cfdfb3dab8daefe13ba1163fc7 "$w/source-v11-lowteacher-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" 7ef182b2e978fadd3cbfe2792251ef9b50823d1c24b51c9141c36aa65ec95bdf "$w/monitor_coworld_steady_interval.py" | sha256sum -c -
printf "%s  %s\n" 54442f9f65a23a38a1a6781d2644770b689b7855fd10e749c042e255308fdd0b "$w/v11-lowteacher-build-15451/build.json" | sha256sum -c -
printf "%s  %s\n" eaa4075cf329dcae53a05e5dcf54a258075a27c33be7d7b1a4b5389da7e4bda5 "$w/v11-lowteacher-build-15451/puffer" | sha256sum -c -
printf "%s  %s\n" 730022aafcb836422d7b97d403d30bcfa6a30552cb6e40e90825a29c4c85ef5f "$w/v11-lowteacher-15451/run/checkpoints/metta_generals/run/0000000004194304.bin" | sha256sum -c -
python3 - "$w/v11-lowteacher-15451/config.json" "$w/v11-lowteacher-15451/run/completed.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
source, completed, target = map(Path, sys.argv[1:])
config = json.loads(source.read_text())
record = json.loads(completed.read_text())
assert config["total_timesteps"] == record["trained_timesteps"] == 4_194_304
assert config["overrides"]["vec.total_agents"] == 4096
assert config["overrides"]["train.minibatch_size"] == 16384
assert config["overrides"]["train.learning_rate"] == 0.0003
assert "checkpoints/metta_generals/run/0000000004194304.bin" in record["checkpoints"]
config["total_timesteps"] = 25_165_824
config["seed"] = 685
config["initialize"] = {
    "run": "/work/v11-lowteacher-15451/run",
    "checkpoint": "/work/v11-lowteacher-15451/run/checkpoints/metta_generals/run/0000000004194304.bin",
    "sha256": "730022aafcb836422d7b97d403d30bcfa6a30552cb6e40e90825a29c4c85ef5f",
}
target.write_text(json.dumps(config) + "\n")
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
    --user "$(id -u):$(id -g)" -v "$w:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work
    -e PYTHONPATH=/work/runtime:/work/source-v11-lowteacher-transfer:/work/source-v11-strongmix:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
ln -s "$out/run" "$w/v11-lowteacher-long-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$w/v11-lowteacher-long-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 22m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-v11-strongmix/integrations/run_puffer_gpu.sh train \
    --config "/work/v11-lowteacher-long-$job/config.json" \
    --build /work/v11-lowteacher-build-15451 \
    --output "/work/v11-lowteacher-long-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval.py" "$w" "$job" "$((now-start))" v11-lowteacher-long; then
        echo "Low-teacher continuation stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

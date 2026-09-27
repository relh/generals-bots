#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-balanced-zero-entropy-134m-paired-validation-v2
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-balanced-zero-entropy-134m-paired-validation-v2-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e75f1862f88847c8893344c8ff26ca0bb35ff284d085ff95b1b000f1c7833d75 \
    "$disk/metta_puffer-balanced-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-balanced-zero-entropy-134m-paired-validation-v2-$job
mkdir -p "$out"
printf "%s  %s\n" 010a7b78afdd24a3c6718cea81bf0bebfe4a5d7407513bf6c84ae013850e02e5 \
    "$disk/evaluate_coworld_frozen_greedy-balanced-20260927.py" | sha256sum -c -
printf "%s  %s\n" af3c1d7fc0c52b8f31a0d9d20d43be36421af5938772810e1116e3de0064527f \
    "$disk/launch_puffer_balanced_greedy-20260927.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
ready=0
for attempt in {1..15}; do
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
printf "GPU_QUERY attempt=%s step_gpus=%s rows=%s\n" "$attempt" "$SLURM_STEP_GPUS" "$usage"
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
echo "gpu_idle_check=$attempt gpu=$gpu memory_mb=$memory utilization=$utilization step_gpus=$SLURM_STEP_GPUS"
if (( memory < 2048 && utilization < 20 )); then
    ready=1
    break
fi
sleep 2
done
(( ready == 1 ))
test -n "$gpu"
name=relh-classic-device-context-balanced-zero-entropy-134m-paired-validation-v2-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for tag in parent zero_entropy; do
    if [[ "$tag" == parent ]]; then
        run=device-context-calibrated-134m-24646
        checkpoint=0000000134217728.bin
        digest=a672058d8b3abfc6d8e8430e88b5431fab786cae2bd909e55c1eb36ce901d1be
    else
        run=device-context-balanced-zero-entropy-134m-25386
        checkpoint=0000000134217728.bin
        digest=$(python3 - "$disk/$run/run" <<"PY"
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
completed = json.loads((run / "completed.json").read_text())
assert completed["trained_timesteps"] == 134217728
assert Path(completed["final_checkpoint"]).name == "0000000134217728.bin"
training = json.loads((run / "training.json").read_text())
assert training["build"]["model_sha256"] == "060fb15a879c290edf54e453a9bf2caea17c610f2d74f7a00e79fd9f613dddf5"
print(hashlib.sha256((run / completed["final_checkpoint"]).read_bytes()).hexdigest())
PY
        )
    fi
    for opponent in expander_harvester sentinel strong_mixed; do
timeout -k 30s 9m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/launch_puffer_balanced_greedy-20260927.py \
    --build /recovery/device-context-calibrated-balanced-aligned-discount-16384-build-25200/build/build.json \
    --run "/recovery/$run/run" --checkpoint "/recovery/$run/run/checkpoints/metta_generals/run/$checkpoint" \
    --sha256 "$digest" --games 512 --pool-size 512 --seed 1103 --opponent "$opponent" \
    --output "/recovery-output/$tag-$opponent" > "$out/$tag-$opponent.log" 2>&1
        cat "$out/$tag-$opponent/evaluation.json"
    done
done

'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-label-free-32768-build
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
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-label-free-32768-build-%j.log
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
out=$disk/device-context-label-free-32768-build-$job
mkdir -p "$out"
python3 - "$disk/device-context-calibrated-16384-build-24499/build/build.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())["config"]
assert config["fabric"]["teacher"]["phases"][0]["coefficient"] == 0.0
config["python_environment"]["spec"]["agents"] = 32768
config["python_environment"]["options"].update(parallel_games=32768, move_hint_scale=.375, split_hint_scale=.125, shaping_gamma=.999, balance_opponent_sides=True)
config["fabric"]["teacher"]["phases"][0]["coefficient"] = 0.0
config["fabric"]["teacher"] = None
config["python_environment"]["spec"]["teacher"] = False
config["python_environment"]["options"]["supervise_teacher"] = False
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
name=relh-classic-device-context-label-free-32768-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config /recovery-output/config.json \
    --output /recovery-output/build > "$out/container.log" 2>&1
printf "%s  %s\n" c074304650ba0f2ef6334ea29274118f1ffa11df49cfc5d6f81f362c1c1810c7 "$disk/verify_zero_teacher_transport-20260927.py" | sha256sum -c -
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery:ro" \
    -v "$out:/recovery-output" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/verify_zero_teacher_transport-20260927.py \
    --original-build /recovery/device-context-calibrated-balanced-aligned-discount-16384-build-25200/build/build.json \
    --build /recovery-output/build/build.json --output /recovery-output/parity.json > "$out/parity.log" 2>&1
cat "$out/parity.json"
python3 - "$out/build/build.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["fabric"]["teacher"] is None
assert manifest["model_state_words"] == 22956
assert manifest["config"]["python_environment"]["options"]["move_hint_scale"] == .375
assert manifest["config"]["python_environment"]["options"]["split_hint_scale"] == .125
assert manifest["config"]["python_environment"]["spec"]["agents"] == 32768
assert manifest["config"]["python_environment"]["options"]["shaping_gamma"] == .999
print("LABEL_FREE_BUILD_OK", manifest["binary_sha256"], manifest["model_sha256"], flush=True)
PY
'

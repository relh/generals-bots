#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-context67m-bundle-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-context67m-bundle-probe-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-context67m-bundle-probe-$job
checkpoint=$disk/device-context-overnight-22922/run/checkpoints/metta_generals/run/0000000067108864.bin
mkdir -p "$out"
printf "%s  %s\n" 8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f "$checkpoint" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" ea5106c8467f10465d39e6a2779cca58fee464f791e1f7f36665041a76357151 \
    "$w/metta_puffer-device-fused.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
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
common=(--rm --gpus "device=$gpu" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v "$disk/neural_codec-context-current.py:/work/source-normalized-16245/integrations/softmax/neural_codec.py:ro"
    -v "$disk/neural_player-context-current.py:/work/source-normalized-16245/integrations/softmax/neural_player.py:ro"
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v "$w/metta_puffer-device-fused.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource)
name=relh-classic-device-context-context67m-bundle-probe-$job
cleanup() { docker stop -t 5 "$name-build" "$name-eval" "$name-deps" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-deps" \
    relh-generals-b300:20260923b python -m pip install --no-cache-dir \
    --target "/recovery/device-context-context67m-bundle-probe-$job/deps" websockets==16.0 \
    > "$out/deps-container.log" 2>&1
timeout -k 30s 8m docker run "${common[@]}" --name "$name-eval" -e JAX_PLATFORMS=cpu \
    -e "PYTHONPATH=/recovery/device-context-context67m-bundle-probe-$job/deps:/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource" \
    relh-generals-b300:20260923b python /recovery/probe_context_frozen_bundle.py \
    --build /work/device-context-eval-22138/build/build.json \
    --checkpoint /recovery/device-context-overnight-22922/run/checkpoints/metta_generals/run/0000000067108864.bin \
    --sha256 8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f \
    --output "/recovery/device-context-context67m-bundle-probe-$job/probe" > "$out/probe-container.log" 2>&1
'

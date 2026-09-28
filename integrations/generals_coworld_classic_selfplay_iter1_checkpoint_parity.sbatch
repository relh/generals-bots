#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-checkpoint-parity
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:08:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-checkpoint-parity-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/classic-selfplay-iter1-checkpoint-parity-$SLURM_JOB_ID
mkdir -p "$out"
source=$disk/audit_coworld_selfplay_checkpoint_parity-v1-20260927.py
printf '%s  %s\n' 55332588304d5d4c64eadf384fd51415cf32701e35d87869ada7ce149aa1e7b7 "$source" | sha256sum -c -
checkpoint=$disk/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin
printf '%s  %s\n' e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf "$checkpoint" | sha256sum -c -
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
name=relh-classic-selfplay-iter1-checkpoint-parity-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
timeout -k 30s 6m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=4 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" \
    -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro" \
    -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery/classic-selfplay-iter1-checkpoint-parity-$SLURM_JOB_ID/cache \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/audit_coworld_selfplay_checkpoint_parity-v1-20260927.py \
    --source-build /recovery/classic-selfplay-teacher-h128-build-27857/build.json \
    --target-build /recovery/classic-selfplay-iter1-build-28048/build/build.json \
    --checkpoint /recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin \
    --sha256 e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf \
    --output "/recovery/classic-selfplay-iter1-checkpoint-parity-$SLURM_JOB_ID/result" \
    > "$out/container.log" 2>&1
cat "$out/container.log" | tail -5
NODE

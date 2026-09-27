#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-transfer-guard
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-transfer-guard-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=2 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
w=/tmp/relh-generals-coworld
printf '%s  %s\n' 8f497fe300eade8653e7fc6b92b23cb0d850cc09d0fddf518de601603da1ea73 "$disk/puffer_selfplay_policy_transfer-v1-20260927.py" | sha256sum -c -
printf '%s  %s\n' 78fbf51cb600f3168d21f99c790f3a6f841a9c6a9ae534139243d414c66e5e13 "$disk/audit_coworld_selfplay_transfer_guard-v1-20260927.py" | sha256sum -c -
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
timeout -k 10s 3m docker run --rm --gpus "device=$gpu" --cpus=2 --ulimit core=0:0 \
    --user "$(id -u):$(id -g)" -v "$disk:/recovery" -v "$w:/work" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b \
    python /recovery/audit_coworld_selfplay_transfer_guard-v1-20260927.py
NODE

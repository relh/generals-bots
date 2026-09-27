#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-pass-breakdown
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-pass-breakdown-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/classic-pass-breakdown-$SLURM_JOB_ID
mkdir -p "$out"
source=$disk/evaluate_coworld_spatial_mixed_frozen-pass-breakdown-20260927.py
printf '%s  %s\n' f3ddf1ce035cc1135d28c954bc6319c7dcb4dbac377ead242c4083ec92079588 "$source" | sha256sum -c -
cp "$source" "$out/evaluate.py"
test -f "$disk/classic-spatial4-hintcurriculum-480-27689/run/completed.json"
test -f "$disk/classic-spatial4-hintcurriculum-480-quality-27703/ppo-sentinel/evaluation.json"
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
name=relh-classic-pass-breakdown-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
run=$disk/classic-spatial4-hintcurriculum-480-27689/run
for step in 4194304 8388608; do
    checkpoint=$(printf '%016d' "$step")
    digest=$(sha256sum "$run/checkpoints/metta_generals/run/$checkpoint.bin" | cut -d' ' -f1)
    result=$out/$checkpoint
    timeout -k 30s 6m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" -v "$disk:/recovery" \
        -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro" \
        -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
        -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
        -v /var/tmp/relh-generals-jax-cache:/cache \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e XDG_CACHE_HOME=/recovery/classic-pass-breakdown-$SLURM_JOB_ID/cache \
        -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
        -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
        relh-generals-b300:20260923b python "/recovery/classic-pass-breakdown-$SLURM_JOB_ID/evaluate.py" \
        --build /recovery/classic-spatial4-hintcurriculum-480-build-27689/build.json \
        --run /recovery/classic-spatial4-hintcurriculum-480-27689/run \
        --checkpoint "/recovery/classic-spatial4-hintcurriculum-480-27689/run/checkpoints/metta_generals/run/$checkpoint.bin" \
        --sha256 "$digest" --games 64 --pool-size 64 --seed 1387 \
        --opponent expander_harvester --force-hint-move --force-hint-split \
        --hint-audit --audit-turns 128 --output "/recovery/classic-pass-breakdown-$SLURM_JOB_ID/$checkpoint" \
        > "$out/$checkpoint.log" 2>&1
    python3 - "$result/evaluation.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
print(json.dumps({'sha256':data['checkpoint_sha256'],'hint_audit':data['hint_audit']},sort_keys=True),flush=True)
PY
done
tar -C "$disk" -czf "$disk/classic-spatial4-pass-breakdown-$SLURM_JOB_ID.tar.gz" \
    "classic-pass-breakdown-$SLURM_JOB_ID" \
    "classic-pass-breakdown-$SLURM_JOB_ID.log"
sha256sum "$disk/classic-spatial4-pass-breakdown-$SLURM_JOB_ID.tar.gz"
NODE

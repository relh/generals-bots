#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-prior05-agreement
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
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-prior05-agreement-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/classic-prior05-agreement-$SLURM_JOB_ID
mkdir -p "$out"
source=$disk/evaluate_coworld_spatial_mixed_frozen-prior05-agreement-20260927.py
printf '%s  %s\n' 70a54e0b816f02e25bd149a3e03140da724242c925583db82ac4211b545c37c6 "$source" | sha256sum -c -
cp "$source" "$out/evaluate.py"
test -f "$disk/classic-spatial4-hintcurriculum-prior05-32m-27745/run/completed.json"
test -f "$disk/classic-spatial4-hintcurriculum-prior05-32m-quality-27760/ppo-sentinel/evaluation.json"
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
name=relh-classic-prior05-agreement-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
run=$disk/classic-spatial4-hintcurriculum-prior05-32m-27745/run
for step in 16777216 33554432; do
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
        -e XDG_CACHE_HOME=/recovery/classic-prior05-agreement-$SLURM_JOB_ID/cache \
        -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
        -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
        relh-generals-b300:20260923b python "/recovery/classic-prior05-agreement-$SLURM_JOB_ID/evaluate.py" \
        --build /recovery/classic-spatial4-hintcurriculum-prior05-32m-build-27745/build.json \
        --run /recovery/classic-spatial4-hintcurriculum-prior05-32m-27745/run \
        --checkpoint "/recovery/classic-spatial4-hintcurriculum-prior05-32m-27745/run/checkpoints/metta_generals/run/$checkpoint.bin" \
        --sha256 "$digest" --games 64 --pool-size 64 --seed 1387 \
        --opponent expander_harvester --force-hint-move --force-hint-split \
        --hint-audit --audit-turns 128 --output "/recovery/classic-prior05-agreement-$SLURM_JOB_ID/$checkpoint" \
        > "$out/$checkpoint.log" 2>&1
    python3 - "$result/evaluation.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
print(json.dumps({'sha256':data['checkpoint_sha256'],'hint_audit':data['hint_audit']},sort_keys=True),flush=True)
PY
done
tar -C "$disk" -czf "$disk/classic-spatial4-prior05-agreement-$SLURM_JOB_ID.tar.gz" \
    "classic-prior05-agreement-$SLURM_JOB_ID" \
    "classic-prior05-agreement-$SLURM_JOB_ID.log"
sha256sum "$disk/classic-spatial4-prior05-agreement-$SLURM_JOB_ID.tar.gz"
NODE

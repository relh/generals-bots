#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-calibrated-h128-agreement
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-calibrated-h128-agreement-%j.log
set -euo pipefail

# Mandatory user policy: verify lowest scheduling priority before workload startup.
slurm_task_priority_record=$(scontrol show job -o "${SLURM_JOB_ID:?}")
[[ " $slurm_task_priority_record " == *" Nice=2147483645 "* ]] || { printf 'Lowest Slurm Nice verification failed\n' >&2; exit 1; }
slurm_task_priority=unknown
for slurm_task_field in $slurm_task_priority_record; do
  case "$slurm_task_field" in Priority=*) slurm_task_priority=${slurm_task_field#Priority=} ;; esac
done
[[ "$slurm_task_priority" =~ ^[0-9]+$ ]] || exit 1
printf 'SLURM_PRIORITY_READBACK JobId=%s Nice=2147483645 Priority=%s\n' "$SLURM_JOB_ID" "$slurm_task_priority" >&2
unset slurm_task_priority_record slurm_task_field

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID
mkdir -p "$out"
source=$disk/evaluate_coworld_spatial_mixed_frozen-counterfactual-20260927.py
printf '%s  %s\n' 8ea17e06f450393dd4b51dc78db3597a8331b6316c4c64e16643fb097b660eff "$source" | sha256sum -c -
cp "$source" "$out/evaluate.py"
test -f "$disk/classic-selfplay-iter1-calibrated-h128-28074/run/completed.json"
test -f "$disk/classic-selfplay-iter1-calibrated-h128-quality-28079/final-sentinel/evaluation.json"
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
name=relh-classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
run=$disk/classic-selfplay-iter1-calibrated-h128-28074/run
for step in 33554432; do
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
        -e XDG_CACHE_HOME=/recovery/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID/cache \
        -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
        -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
        relh-generals-b300:20260923b python "/recovery/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID/evaluate.py" \
        --build /recovery/classic-selfplay-iter1-build-28048/build/build.json \
        --run /recovery/classic-selfplay-iter1-calibrated-h128-28074/run \
        --checkpoint "/recovery/classic-selfplay-iter1-calibrated-h128-28074/run/checkpoints/metta_generals/run/$checkpoint.bin" \
        --sha256 "$digest" --games 64 --pool-size 64 --seed 1387 \
        --opponent expander_harvester --force-hint-move --force-hint-split \
        --hint-audit --audit-turns 128 --output "/recovery/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID/$checkpoint" \
        > "$out/$checkpoint.log" 2>&1
    python3 - "$result/evaluation.json" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))
print(json.dumps({'sha256':data['checkpoint_sha256'],'hint_audit':data['hint_audit']},sort_keys=True),flush=True)
PY
done
tar -C "$disk" -czf "$disk/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID.tar.gz" \
    "classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID" \
    "classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID.log"
sha256sum "$disk/classic-selfplay-iter1-calibrated-h128-agreement-$SLURM_JOB_ID.tar.gz"
NODE

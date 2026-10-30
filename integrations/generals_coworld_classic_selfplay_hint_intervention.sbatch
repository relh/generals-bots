#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-hint-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-hint-audit-%j.log
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
out=$disk/classic-selfplay-hint-audit-$SLURM_JOB_ID
build=$disk/classic-selfplay-prior2-v1-build-27520/build.json
run=$disk/classic-selfplay-prior2-v1-27520/run
baseline=$disk/classic-selfplay-prior2-v1-quality-27521
mkdir -p "$out"
printf '%s  %s\n' e51c64c3b89b1307945fdd3c923de688a79ee1786735b66e0973ac4000d143db "$disk/evaluate_coworld_spatial_mixed-fabric-hint-audit-v1.py" | sha256sum -c -
printf '%s  %s\n' 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
python3 - "$build" "$run" "$baseline" "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
build, run, baseline, out = map(Path, sys.argv[1:])
record = json.loads((run / 'training.json').read_text())
assert record['build'] == json.loads(build.read_text())
checkpoint = run / 'checkpoints/metta_generals/run/0000000016777216.bin'
assert checkpoint.is_file()
assert all((baseline / f'final-{opponent}/outcomes.npy').is_file()
           for opponent in ('expander_harvester', 'sentinel'))
(out / 'identity.json').write_text(json.dumps({
    'checkpoint': str(checkpoint),
    'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
}) + '\n')
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
printf 'allocated_gpus=%s visible_gpus=%s usage=%s\n' "$SLURM_STEP_GPUS" "${#rows[@]}" "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
checkpoint=$(python3 - "$out/identity.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['checkpoint'])
PY
)
digest=$(python3 - "$out/identity.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['sha256'])
PY
)
name=relh-classic-selfplay-hint-audit-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
for opponent in expander_harvester sentinel; do
    timeout -k 30s 7m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
        --ulimit core=0:0 --user "$(id -u):$(id -g)" \
        -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output" \
        -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro" \
        -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
        -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
        -v /var/tmp/relh-generals-jax-cache:/cache \
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e XDG_CACHE_HOME=/recovery-output/cache \
        -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
        -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
        relh-generals-b300:20260923b python /recovery/evaluate_coworld_spatial_mixed-fabric-hint-audit-v1.py \
        --build /recovery/classic-selfplay-prior2-v1-build-27520/build.json \
        --run /recovery/classic-selfplay-prior2-v1-27520/run \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
        --force-hint-move --force-hint-split \
        --output "/recovery-output/$opponent" > "$out/$opponent.log" 2>&1
done
python3 - "$baseline" "$out" <<'PY'
import json, sys
from pathlib import Path
baseline, out = map(Path, sys.argv[1:])
results = {}
for opponent in ('expander_harvester', 'sentinel'):
    before = (baseline / f'final-{opponent}/outcomes.npy').read_bytes()
    after = (out / opponent / 'outcomes.npy').read_bytes()
    assert before == after, 'Forced hint changed at least one game outcome'
    prior = json.loads((baseline / f'final-{opponent}/evaluation.json').read_text())
    record = json.loads((out / opponent / 'evaluation.json').read_text())
    results[opponent] = dict(
        network_wld=[prior['wins'], prior['losses'], prior['draws']],
        hint_wld=[record['wins'], record['losses'], record['draws']],
        outcome_arrays_equal=True,
        intervention=record['intervention'],
    )
(out / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results), flush=True)
PY
NODE

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-prior2-residual-ablation
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:18:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-prior2-residual-ablation-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
source=$disk/ablation-source-20260927/integrations
out=$disk/classic-prior2-residual-ablation-$SLURM_JOB_ID
build=$disk/classic-selfplay-prior2-v1-build-27520/build.json
run=$disk/classic-selfplay-prior2-v1-27520/run
baseline=$disk/classic-selfplay-prior2-v1-quality-27521/final-expander_harvester
mkdir -p "$out"
printf '%s  %s\n' 67070f0752bb652d2b94bbe0a4a341ebebec0a8e82753d3cf3e7eccadc14e37b "$source/make_fabric_hint_prior_ablation.py" | sha256sum -c -
printf '%s  %s\n' f95ce4b94bd60c291ba21c986be1fe528ccdd202e3454747f1c836bb93a56abc "$source/evaluate_coworld_spatial_mixed_frozen.py" | sha256sum -c -
printf '%s  %s\n' 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
python3 - "$build" "$run" "$baseline" <<'PY'
import json,sys
from pathlib import Path
build,run,baseline=map(Path,sys.argv[1:])
assert json.loads((run/'training.json').read_text())['build']==json.loads(build.read_text())
assert (run/'completed.json').is_file()
assert (run/'checkpoints/metta_generals/run/0000000016777216.bin').is_file()
assert (baseline/'outcomes.npy').is_file()
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-prior2-residual-ablation-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
common=(--rm --name "$name" --gpus "device=$gpu" --cpus=8
    --ulimit core=0:0 --user "$(id -u):$(id -g)"
    -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output"
    -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro"
    -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e XDG_CACHE_HOME=/recovery-output/cache
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
timeout -k 30s 8m docker run "${common[@]}" relh-generals-b300:20260923b \
    python /recovery/ablation-source-20260927/integrations/make_fabric_hint_prior_ablation.py \
    --build /recovery/classic-selfplay-prior2-v1-build-27520/build.json \
    --checkpoint /recovery/classic-selfplay-prior2-v1-27520/run/checkpoints/metta_generals/run/0000000016777216.bin \
    --sha256 9c270c0c8e649356ed9041ab9231c86b962dcda669c492a4aa092bba3fd99fd4 \
    --target-strength 0.5 --output /recovery-output/ablation > "$out/prepare.log" 2>&1
digest=$(python3 - "$out/ablation/ablation.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['ablation_checkpoint_sha256'])
PY
)
timeout -k 30s 6m docker run "${common[@]}" relh-generals-b300:20260923b \
    python /recovery/ablation-source-20260927/integrations/evaluate_coworld_spatial_mixed_frozen.py \
    --build /recovery/classic-selfplay-prior2-v1-build-27520/build.json \
    --run /recovery/classic-selfplay-prior2-v1-27520/run \
    --checkpoint /recovery-output/ablation/checkpoint.bin --sha256 "$digest" \
    --diagnostic-checkpoint --games 128 --pool-size 128 --seed 1386 \
    --opponent expander_harvester --output /recovery-output/expander_harvester \
    > "$out/expander_harvester.log" 2>&1
python3 - "$baseline" "$out" <<'PY'
import json,sys
from pathlib import Path
baseline,out=map(Path,sys.argv[1:])
prior=json.loads((baseline/'evaluation.json').read_text())
result=json.loads((out/'expander_harvester/evaluation.json').read_text())
record={
    'kind':'frozen_hint_prior_residual_ablation',
    'baseline_wld':[prior['wins'],prior['losses'],prior['draws']],
    'ablation_wld':[result['wins'],result['losses'],result['draws']],
    'outcome_arrays_equal':(baseline/'outcomes.npy').read_bytes()==(out/'expander_harvester/outcomes.npy').read_bytes(),
    'ablation_checkpoint_sha256':result['checkpoint_sha256'],
}
(out/'comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
NODE

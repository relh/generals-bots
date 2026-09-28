#!/usr/bin/env bash
# Run on the submit host after replacing __EVALUATOR_SOURCE__ with base64 source.
# Uses an existing training allocation; creates no new job.
set -euo pipefail
job=${1:?training job ID required}
step=${2:?saved checkpoint step required}
[[ "$job" =~ ^[0-9]+$ && "$step" =~ ^[0-9]+$ ]]
export GENERALS_LIVE_CHECKPOINT_STEP=$step
srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=2 --time=00:12:00 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
parent=$disk/classic-flat-scripted-potential-pilot-29432
source=$parent/staged
step=${GENERALS_LIVE_CHECKPOINT_STEP:?}
out=$disk/classic-flat-potential-300m-pilot-$SLURM_JOB_ID
checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
test -f "$checkpoint.learner.json"
evalout=$out/live-$step
test ! -e "$evalout"
mkdir "$evalout"
base64 -d <<'SOURCE' > "$evalout/evaluate.py"
__EVALUATOR_SOURCE__
SOURCE
sha256sum "$evalout/evaluate.py" "$checkpoint"
gpu=$(bash "$parent/staged/allocated_gpu_uuid.sh")
echo "live evaluation job=$SLURM_JOB_ID step=$step gpu_uuid=$gpu node=$(hostname)"
nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader
name=relh-classic-live-quality-$SLURM_JOB_ID-$step
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=2 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$source/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$source/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
for opponent in expander_harvester sentinel; do
    timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
        python "/recovery/classic-flat-potential-300m-pilot-$SLURM_JOB_ID/live-$step/evaluate.py" \
        --build "/recovery/classic-flat-potential-300m-pilot-$SLURM_JOB_ID/build/build.json" \
        --run "/recovery/classic-flat-potential-300m-pilot-$SLURM_JOB_ID/run" \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" --in-progress-checkpoint \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
        --output "/recovery/classic-flat-potential-300m-pilot-$SLURM_JOB_ID/live-$step/$opponent" \
        > "$evalout/$opponent.log" 2>&1
    python3 - "$evalout/$opponent/evaluation.json" <<'SCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
SCORE
done
NODE

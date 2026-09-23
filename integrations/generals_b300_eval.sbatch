#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-gpu-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/eval-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
export RUN_NAME=${RUN_NAME:-gpu-batch-pilot-v3}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-}
export OUTPUT_NAME=${OUTPUT_NAME:-$RUN_NAME-evaluation}
export EVAL_CONFIG=${EVAL_CONFIG:-gpu-batch-eval.json}
test -f "$workspace/$RUN_NAME/run.json"
if [[ -z "$CHECKPOINT_PATH" ]]; then
    test -f "$workspace/$RUN_NAME/completed.json"
else
    [[ "$CHECKPOINT_PATH" == /work/* ]]
    test -s "$workspace/${CHECKPOINT_PATH#/work/}"
fi

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    docker run --rm --gpus "device=${SLURM_STEP_GPUS}" \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -w /work \
        -e HOME=/work \
        -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash /work/source-gpu/integrations/eval_puffer_gpu.sh \
        "/work/$RUN_NAME" \
        /work/build-gpu-batch-v5 \
        "/work/$OUTPUT_NAME" \
        "/work/$EVAL_CONFIG" \
        "$CHECKPOINT_PATH"
'

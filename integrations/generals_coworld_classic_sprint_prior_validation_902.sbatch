#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-coworld-sprint-prior-validation-902
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/sprint-prior-eval-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-coworld
test -d "$workspace/sprint-prior-build-12344"
test -s "$workspace/sprint-prior-pilot-12344-0/completed.json"
test -d /tmp/relh-generals-gpu/runtime
test -d /tmp/relh-generals-gpu/source-gpu/metta_training

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-coworld
    name="relh-generals-coworld-sprint-prior-validation-902-${SLURM_JOB_ID}"
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
            sleep 1
        done
    ) > "$workspace/gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    nvidia-smi -L
    timeout -k 30s 9m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource \
        relh-generals-b300:20260923b bash -lc "
            set -euo pipefail
            bash /work/source/integrations/eval_puffer_gpu.sh \
                /work/sprint-prior-pilot-12344-0 \
                /work/sprint-prior-build-12344 \
                /work/sprint-prior-validation-902-${SLURM_JOB_ID} \
                /work/source/integrations/configs/gpu-coworld-classic-validation-seed902.json
        "
'

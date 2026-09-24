#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-coworld-expander-validation
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
#SBATCH --output=/tmp/relh-generals-coworld/expander-eval-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-coworld
: "${TRAIN_JOB_ID:?Pass the completed GPU pilot job id through sbatch --export}"
seed="${VALIDATION_SEED:-901}"
case "$seed" in
    901) validation_config=gpu-coworld-classic-validation.json ;;
    902) validation_config=gpu-coworld-classic-validation-seed902.json ;;
    *) echo "Unsupported held-out validation seed: $seed" >&2; exit 2 ;;
esac
export TRAIN_JOB_ID seed validation_config
test -s "$workspace/expander-prior-pilot-${TRAIN_JOB_ID}-0/completed.json"
test -s "$workspace/expander-prior-build-${TRAIN_JOB_ID}/build.json"

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-coworld
    name="relh-generals-coworld-expander-validation-${SLURM_JOB_ID}"
    cleanup() {
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 9m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource \
        relh-generals-b300:20260923b bash /work/source/integrations/eval_puffer_gpu.sh \
            "/work/expander-prior-pilot-${TRAIN_JOB_ID}-0" \
            "/work/expander-prior-build-${TRAIN_JOB_ID}" \
            "/work/expander-prior-validation-${seed}-${SLURM_JOB_ID}" \
            "/work/source/integrations/configs/${validation_config}"
'

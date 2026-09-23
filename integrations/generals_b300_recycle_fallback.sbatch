#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-gpu-recycle
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/recycle-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
source_root="$workspace/source-recycle"
evaluation="$workspace/gpu-batch-30m-evaluation/evaluation.json"
test -f "$evaluation"
test -f "$source_root/integrations/metta_puffer.py"
test -f "$workspace/runtime/jax/__init__.py"

# The dependency on the original held-out evaluation is set at submission.
# Skip the second 30M-step run if that policy already reaches the target.
if python3 - "$evaluation" <<'PY'
import json
import statistics
import sys

with open(sys.argv[1]) as source:
    results = json.load(source)["results"]
if len(results) != 5 or any(result["games"] != 16 for result in results):
    print("Expected five held-out seeds with sixteen batch episodes each", file=sys.stderr)
    sys.exit(2)
performance = statistics.mean(result["perf"] for result in results)
print(f"Original held-out performance: {performance:.6f}", flush=True)
sys.exit(0 if performance >= 0.6 else 3)
PY
then
    echo "Original policy passed the held-out threshold; skipping fallback."
    exit 0
else
    status=$?
    if [[ "$status" != 3 ]]; then
        echo "Cannot trust the original evaluation; not starting fallback." >&2
        exit "$status"
    fi
fi

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    docker run --rm --gpus "device=${SLURM_STEP_GPUS}" \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -w /work \
        -e HOME=/work \
        -e PYTHONPATH=/work/runtime:/work/source-recycle \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-recycle/integrations/run_puffer_gpu.sh build \
                --config /work/source-recycle/integrations/configs/gpu-batch-build.json \
                --output /work/build-gpu-recycle-v1
            bash /work/source-recycle/integrations/run_puffer_gpu.sh train \
                --config /work/source-recycle/integrations/configs/gpu-batch-30m.json \
                --build /work/build-gpu-recycle-v1 \
                --output /work/gpu-batch-recycle-30m
            bash /work/source-recycle/integrations/eval_puffer_gpu.sh \
                /work/gpu-batch-recycle-30m \
                /work/build-gpu-recycle-v1 \
                /work/gpu-batch-recycle-30m-evaluation \
                /work/source-recycle/integrations/configs/gpu-batch-eval.json
        "
'

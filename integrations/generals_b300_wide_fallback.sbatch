#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-gpu-wide
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/wide-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
source_root="$workspace/source-wide"
evaluation="$workspace/gpu-batch-recycle-30m-evaluation/evaluation.json"
test -f "$evaluation"
test -f "$source_root/integrations/metta_puffer.py"
test -f "$source_root/metta_training/cli.py"
test -f "$source_root/fabric/__init__.py"
test -s "$source_root/integrations/configs/gpu-batch-wide-build.json"
test -s "$source_root/integrations/configs/gpu-batch-30m.json"
test -s "$source_root/integrations/configs/gpu-batch-eval.json"
test -f "$workspace/runtime/jax/__init__.py"

# Submit with afterok on the recycling job, which writes the held-out evaluation.
# Skip this wider 30M-step run if the recycling policy reaches the target.
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
print(f"Recycling held-out performance: {performance:.6f}", flush=True)
sys.exit(0 if performance >= 0.6 else 3)
PY
then
    echo "Recycling policy passed the held-out threshold; skipping wide run."
    exit 0
else
    status=$?
    if [[ "$status" != 3 ]]; then
        echo "Cannot trust the recycling evaluation; not starting wide run." >&2
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
        -e PYTHONPATH=/work/runtime:/work/source-wide \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-wide/integrations/run_puffer_gpu.sh build \
                --config /work/source-wide/integrations/configs/gpu-batch-wide-build.json \
                --output /work/build-gpu-wide-v1
            bash /work/source-wide/integrations/run_puffer_gpu.sh train \
                --config /work/source-wide/integrations/configs/gpu-batch-30m.json \
                --build /work/build-gpu-wide-v1 \
                --output /work/gpu-batch-wide-30m
            bash /work/source-wide/integrations/eval_puffer_gpu.sh \
                /work/gpu-batch-wide-30m \
                /work/build-gpu-wide-v1 \
                /work/gpu-batch-wide-30m-evaluation \
                /work/source-wide/integrations/configs/gpu-batch-eval.json
        "
'

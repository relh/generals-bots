#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-coworld-expander-stability
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/stability-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-coworld
test -s "$workspace/source/integrations/configs/gpu-coworld-classic-expander-prior-build.json"
test -s "$workspace/source/integrations/configs/gpu-coworld-classic-sparse-dual-pilot.json"
for index in 0 1; do
    seed=$((601 + index))
    config="$workspace/expander-stability-pilot-${SLURM_JOB_ID}-${index}.json"
    python3 - "$workspace/source/integrations/configs/gpu-coworld-classic-sparse-dual-pilot.json" "$config" "$seed" <<'PY'
import json
import sys
from pathlib import Path
source, destination, seed = sys.argv[1:]
config = json.loads(Path(source).read_text())
config["seed"] = int(seed)
config["total_timesteps"] = 3_145_728
config["overrides"]["train.learning_rate"] = 0.003
config["overrides"]["base.checkpoint_interval"] = 24
Path(destination).write_text(json.dumps(config) + "\n")
PY
done

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-coworld
    prefix="relh-generals-coworld-expander-stability-${SLURM_JOB_ID}"
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi -i 0 --query-gpu=utilization.gpu,utilization.memory,memory.used \
                --format=csv,noheader,nounits
            sleep 1
        done
    ) > "$workspace/expander-stability-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "${prefix}-build" "${prefix}-0" "${prefix}-1" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    common=(--rm --gpus "device=${SLURM_STEP_GPUS}" --ulimit core=0:0
        --user "$(id -u):$(id -g)" -v "$workspace:/work"
        -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource)
    timeout -k 30s 3m docker run "${common[@]}" --name "${prefix}-build" --cpus=16 \
        relh-generals-b300:20260923b bash /work/source/integrations/run_puffer_gpu.sh build \
        --config /work/source/integrations/configs/gpu-coworld-classic-expander-prior-build.json \
        --output "/work/expander-stability-build-${SLURM_JOB_ID}"
    for index in 0 1; do
        timeout -k 30s 15m docker run "${common[@]}" --name "${prefix}-${index}" --cpus=8 \
            relh-generals-b300:20260923b bash /work/source/integrations/run_puffer_gpu.sh train \
            --config "/work/expander-stability-pilot-${SLURM_JOB_ID}-${index}.json" \
            --build "/work/expander-stability-build-${SLURM_JOB_ID}" \
            --output "/work/expander-stability-pilot-${SLURM_JOB_ID}-${index}" \
            > "$workspace/expander-stability-pilot-${SLURM_JOB_ID}-${index}.log" 2>&1 &
        pids[$index]=$!
    done
    started=$(date +%s)
    while kill -0 "${pids[0]}" 2>/dev/null || kill -0 "${pids[1]}" 2>/dev/null; do
        sleep 15
        if ! python3 "$workspace/source/integrations/monitor_coworld_long.py" \
            "$workspace" "$SLURM_JOB_ID" "$(( $(date +%s) - started ))" expander-stability; then
            echo "Classic two-trainer guard stopped the run" >&2
            cleanup
            for pid in "${pids[@]}"; do wait "$pid" || true; done
            exit 1
        fi
    done
    status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    exit "$status"
'

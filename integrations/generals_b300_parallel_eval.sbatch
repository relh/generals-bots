#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-parallel-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/parallel-eval-%j.log
set -euo pipefail

export EVAL_RUN_JOB_ID=${EVAL_RUN_JOB_ID:?training Slurm job ID required}
export EVAL_INDEXES=${EVAL_INDEXES:-0,1,2,3}
export EVAL_SEEDS=${EVAL_SEEDS:-901,902,903,904,905}
export EVAL_EPISODES=${EVAL_EPISODES:-8}
export EVAL_PARALLELISM=${EVAL_PARALLELISM:-4}
export EVAL_PREFIX=${EVAL_PREFIX:-gpu-batch-four-long-${EVAL_RUN_JOB_ID}-validation}
export EVAL_BUILD_NAME=${EVAL_BUILD_NAME:-build-gpu-wide16x16-6021}
export EVAL_SOURCE_NAME=${EVAL_SOURCE_NAME:-source-gpu}
export EVAL_ALLOW_TRANSFER=${EVAL_ALLOW_TRANSFER:-false}
test -f "/tmp/relh-generals-gpu/$EVAL_BUILD_NAME/build.json"
test -f "/tmp/relh-generals-gpu/$EVAL_SOURCE_NAME/integrations/eval_puffer_gpu.sh"

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=64 --cpu-bind=cores bash -s <<'RUN'
set -euo pipefail
workspace=/tmp/relh-generals-gpu
IFS=, read -ra indexes <<< "$EVAL_INDEXES"
IFS=, read -ra seeds <<< "$EVAL_SEEDS"
[[ "$EVAL_EPISODES" =~ ^[0-9]+$ && "$EVAL_EPISODES" -gt 0 ]]
[[ "$EVAL_PARALLELISM" =~ ^[0-9]+$ && "$EVAL_PARALLELISM" -ge 1 && "$EVAL_PARALLELISM" -le 16 ]]
(( 64 % EVAL_PARALLELISM == 0 ))
cpus_per_eval=$((64 / EVAL_PARALLELISM))
[[ "$EVAL_ALLOW_TRANSFER" == true || "$EVAL_ALLOW_TRANSFER" == false ]]
for index in "${indexes[@]}"; do
    [[ "$index" =~ ^[0-3]$ ]]
    test -f "$workspace/gpu-batch-four-long-${EVAL_RUN_JOB_ID}-${index}/completed.json"
done
for seed in "${seeds[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]]
    python3 - "$workspace/eval-parallel-${SLURM_JOB_ID}-seed${seed}.json" "$seed" "$EVAL_EPISODES" "$EVAL_ALLOW_TRANSFER" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "seeds": [int(sys.argv[2])],
    "episodes_per_seed": int(sys.argv[3]),
    "timeout_seconds": 600,
    "allow_environment_transfer": sys.argv[4] == "true",
}))
PY
done
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits || true
        sleep 5
    done
) > "$workspace/parallel-eval-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
sampler=$!
containers=()
cleanup() {
    kill "$sampler" 2>/dev/null || true
    for container in "${containers[@]}"; do
        docker stop -t 2 "$container" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
pids=()
for seed in "${seeds[@]}"; do
    for index in "${indexes[@]}"; do
        name="relh-generals-parallel-eval-${SLURM_JOB_ID}-${index}-${seed}"
        containers+=("$name")
        (
            timeout -k 15s 15m docker run --rm --name "$name" \
                --gpus "device=${SLURM_STEP_GPUS}" --cpus="$cpus_per_eval" \
                --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
                -e HOME=/work -e "PYTHONPATH=/work/runtime:/work/${EVAL_SOURCE_NAME}" \
                relh-generals-b300:20260923b \
                bash "/work/${EVAL_SOURCE_NAME}/integrations/eval_puffer_gpu.sh" \
                "/work/gpu-batch-four-long-${EVAL_RUN_JOB_ID}-${index}" \
                "/work/${EVAL_BUILD_NAME}" \
                "/work/${EVAL_PREFIX}-policy${index}-seed${seed}" \
                "/work/eval-parallel-${SLURM_JOB_ID}-seed${seed}.json" \
                > "$workspace/${EVAL_PREFIX}-policy${index}-seed${seed}.log" 2>&1
        ) &
        pids+=("$!")
        if (( ${#pids[@]} == EVAL_PARALLELISM )); then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 - <<'PY'
import json
import os
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
indexes = [int(value) for value in os.environ["EVAL_INDEXES"].split(",")]
seeds = [int(value) for value in os.environ["EVAL_SEEDS"].split(",")]
episodes = int(os.environ["EVAL_EPISODES"])
prefix = os.environ["EVAL_PREFIX"]
summary = {"training_job_id": int(os.environ["EVAL_RUN_JOB_ID"]), "seeds": seeds, "policies": {}}
for index in indexes:
    results = []
    hashes = set()
    for seed in seeds:
        path = workspace / f"{prefix}-policy{index}-seed{seed}" / "evaluation.json"
        evaluation = json.loads(path.read_text())
        assert evaluation["config"]["seeds"] == [seed]
        assert evaluation["config"]["episodes_per_seed"] == episodes
        assert len(evaluation["results"]) == 1
        result = evaluation["results"][0]
        assert result["seed"] == seed and result["games"] >= episodes
        hashes.add(evaluation["checkpoint_sha256"])
        results.append(result)
    assert len(hashes) == 1
    games = sum(result["games"] for result in results)
    perf = sum(result["perf"] * result["games"] for result in results) / games
    summary["policies"][str(index)] = {
        "checkpoint_sha256": hashes.pop(),
        "games": games,
        "perf": perf,
        "results": results,
    }
destination = workspace / f"{prefix}-summary.json"
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({index: item["perf"] for index, item in summary["policies"].items()}), flush=True)
print(destination, flush=True)
PY
RUN

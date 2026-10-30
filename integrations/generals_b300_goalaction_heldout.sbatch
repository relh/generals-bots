#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-goalaction-heldout
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:18:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/goalaction-heldout-%j.log
set -euo pipefail

export TRAIN_JOB_ID=${TRAIN_JOB_ID:?training Slurm job ID required}
export VALIDATION_JOB_ID=${VALIDATION_JOB_ID:?validation Slurm job ID required}
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    test -s "$workspace/goalaction-two-30m-${TRAIN_JOB_ID}-0/completed.json"
    test -s "$workspace/goalaction-two-30m-${TRAIN_JOB_ID}-1/completed.json"
    test -s "$workspace/build-goalaction-validation-${VALIDATION_JOB_ID}/build.json"
    index=$(python3 - <<"PY"
import hashlib
import json
import os
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
training = os.environ["TRAIN_JOB_ID"]
validation = os.environ["VALIDATION_JOB_ID"]
job = os.environ["SLURM_JOB_ID"]
scores = []
for index in range(2):
    run = workspace / f"goalaction-two-30m-{training}-{index}"
    completed = json.loads((run / "completed.json").read_text())
    assert completed["trained_timesteps"] == 31_457_280
    checkpoint = run / completed["final_checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    suffix = "" if index == 0 else "-1"
    evaluation = json.loads((workspace / f"goalaction-validation-{validation}{suffix}/evaluation.json").read_text())
    assert evaluation["checkpoint_sha256"] == digest
    assert evaluation["config"]["seeds"] == [901, 902]
    assert evaluation["config"]["episodes_per_seed"] == 1
    assert evaluation["config"]["allow_environment_transfer"] is True
    results = evaluation["results"]
    assert [item["seed"] for item in results] == [901, 902]
    games = sum(item["games"] for item in results)
    score = sum(item["perf"] * item["games"] for item in results) / games
    scores.append({"index": index, "perf": score, "games": games, "checkpoint_sha256": digest})
selected = max(scores, key=lambda item: (item["perf"], -item["index"]))
assert selected["perf"] >= 0.60, "validation below 0.60; keep held-out seeds sealed"
(workspace / f"goalaction-heldout-selection-{job}.json").write_text(json.dumps({
    "training_job_id": training, "validation_job_id": validation,
    "validation_seeds": [901, 902], "candidates": scores,
    "selected_index": selected["index"],
}, indent=2))
(workspace / f"goalaction-heldout-{job}.json").write_text(json.dumps({
    "seeds": [1001, 1002, 1003, 1004, 1005],
    "episodes_per_seed": 1,
    "timeout_seconds": 900,
    "allow_environment_transfer": True,
}))
print(selected["index"])
PY
    )
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/goalaction-heldout-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-goalaction-heldout-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 16m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-gpu/integrations/eval_puffer_gpu.sh \\
                /work/goalaction-two-30m-${TRAIN_JOB_ID}-${index} \\
                /work/build-goalaction-validation-${VALIDATION_JOB_ID} \\
                /work/goalaction-heldout-${SLURM_JOB_ID} \\
                /work/goalaction-heldout-${SLURM_JOB_ID}.json
        "
'

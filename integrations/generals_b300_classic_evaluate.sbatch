#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-classic-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/classic-eval-%j.log
set -euo pipefail

workspace=/tmp/relh-generals-gpu
source_root="$workspace/source-classic10"
pilot_job_id=${PILOT_JOB_ID:?evaluation pilot job ID required}
train_job_id=${TRAIN_JOB_ID:?training job ID required}
pilot_summary="$workspace/gpu-batch-four-long-${train_job_id}-validation-summary.json"
pilot_gpu="$workspace/parallel-eval-gpu-${pilot_job_id}.csv"
test -s "$pilot_summary"
test -s "$pilot_gpu"
test -s "$source_root/integrations/configs/gpu-batch-classic10-build.json"

# The short pilot is run first. Stop here if eight parallel evaluators still
# leave the GPU mostly idle; this job then releases its allocation immediately.
python3 - "$pilot_summary" "$pilot_gpu" <<'PY'
import json
import statistics
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
assert set(summary["policies"]) == {"0", "1", "2", "3"}
games = sum(policy["games"] for policy in summary["policies"].values())
assert games >= 16
samples = []
for line in Path(sys.argv[2]).read_text().splitlines():
    fields = line.split(",")
    if len(fields) >= 3:
        try:
            samples.append(int(fields[0].strip()))
        except ValueError:
            pass
active = samples[5:-2]
if len(active) < 5:
    raise SystemExit(f"Pilot too short for utilization audit: {len(samples)} GPU samples")
utilization = statistics.median(active)
print(f"Evaluation pilot: {games} games, {len(samples)} GPU samples, "
      f"median active utilization {utilization:.1f}%", flush=True)
if utilization < 50:
    raise SystemExit("Evaluation GPU utilization below 50%; releasing allocation")
PY

build_name="build-classic10-${SLURM_JOB_ID}"
test ! -e "$workspace/$build_name"
export build_name
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=64 --cpu-bind=cores \
    bash -s <<'RUN'
set -euo pipefail
docker run --rm --name "relh-generals-classic-build-${SLURM_JOB_ID}" \
    --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-gpu:/work -w /work \
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-classic10 \
    relh-generals-b300:20260923b \
    bash /work/source-classic10/integrations/run_puffer_gpu.sh build \
    --config /work/source-classic10/integrations/configs/gpu-batch-classic10-build.json \
    --output "/work/$build_name"
RUN

export EVAL_RUN_JOB_ID="$train_job_id"
export EVAL_BUILD_NAME="$build_name"
export EVAL_SOURCE_NAME=source-classic10
export EVAL_ALLOW_TRANSFER=true
export EVAL_PARALLELISM=8
export EVAL_INDEXES=0,1,2,3
export EVAL_SEEDS=901,902,903,904,905
export EVAL_EPISODES=8
for checkpoint in 20480000 25600000 final; do
    case "$checkpoint" in
        20480000) label=20; export EVAL_CHECKPOINT_STEP="$checkpoint" ;;
        25600000) label=25; export EVAL_CHECKPOINT_STEP="$checkpoint" ;;
        final) label=final; export EVAL_CHECKPOINT_STEP=0 ;;
    esac
    export EVAL_PREFIX="gpu-batch-four-long-${train_job_id}-classic-v${label}"
    bash "$source_root/integrations/generals_b300_parallel_eval.sbatch"
done

selection=$(python3 - "$workspace" "$train_job_id" <<'PY'
import json
import sys
from pathlib import Path

workspace, job_id = Path(sys.argv[1]), sys.argv[2]
candidates = []
for label in ("20", "25", "final"):
    path = workspace / f"gpu-batch-four-long-{job_id}-classic-v{label}-summary.json"
    summary = json.loads(path.read_text())
    assert summary["seeds"] == [901, 902, 903, 904, 905]
    assert set(summary["policies"]) == {"0", "1", "2", "3"}
    for index, policy in summary["policies"].items():
        assert policy["games"] >= 40
        candidates.append((policy["perf"], index, policy["checkpoint_step"] or 0))
_, index, step = max(candidates, key=lambda item: (item[0], -int(item[1]), item[2]))
print(index, step)
PY
)
read -r selected checkpoint_step <<< "$selection"
echo "Selected policy $selected at step ${checkpoint_step:-final} from classic validation"
export EVAL_INDEXES="$selected"
export EVAL_CHECKPOINT_STEP="$checkpoint_step"
export EVAL_SEEDS=1001,1002,1003,1004,1005
export EVAL_EPISODES=16
export EVAL_PREFIX="gpu-batch-four-long-${train_job_id}-classic-heldout"
bash "$source_root/integrations/generals_b300_parallel_eval.sbatch"

python3 - "$workspace/${EVAL_PREFIX}-summary.json" "$selected" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
selected = sys.argv[2]
policy = summary["policies"][selected]
assert summary["seeds"] == [1001, 1002, 1003, 1004, 1005]
assert policy["games"] >= 80
print(f"Classic held-out policy {selected}: {policy['perf']:.6f} "
      f"across {policy['games']} games; target >= 0.60", flush=True)
PY

proof="$workspace/gpu-batch-four-long-${train_job_id}-classic-proof.json"
if python3 "$workspace/verify_classic_heldout.py" "$workspace" "$train_job_id" --output "$proof"; then
    echo "Independent classic held-out audit passed."
else
    status=$?
    [[ "$status" == 3 ]] || exit "$status"
    echo "Independent audit confirmed a score below 0.60; Sentinel smoke may run."
fi
# Allow the mettabox watcher to copy the proof before this allocation exits.
sleep 15

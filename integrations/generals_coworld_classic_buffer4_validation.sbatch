#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-buffer4-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/buffer4-eval-%j.log
set -euo pipefail

if (( $# )); then
    if (( $# != 4 )); then
        echo "Usage: sbatch $0 TRAIN_JOB_ID CHECKPOINT_STEP VALIDATION_SEED EPISODES_PER_SEED" >&2
        exit 2
    fi
    TRAIN_JOB_ID=$1
    CHECKPOINT_STEP=$2
    VALIDATION_SEED=$3
    EPISODES_PER_SEED=$4
fi
: "${TRAIN_JOB_ID:?Set TRAIN_JOB_ID for a completed four-buffer run}"
: "${CHECKPOINT_STEP:?Set CHECKPOINT_STEP to a checkpoint step or final}"
: "${VALIDATION_SEED:?Set VALIDATION_SEED to a held-out seed}"
EPISODES_PER_SEED=${EPISODES_PER_SEED:-1}
workspace=/tmp/relh-generals-coworld
run="$workspace/buffer4-long-${TRAIN_JOB_ID}/run"
build="$workspace/expander-quad-stability-build-13219"
output="$workspace/buffer4-eval-${TRAIN_JOB_ID}-${CHECKPOINT_STEP}-${VALIDATION_SEED}-${SLURM_JOB_ID}"
request="$output.request"

python3 - "$run" "$build" "$request" "$CHECKPOINT_STEP" "$VALIDATION_SEED" "$EPISODES_PER_SEED" <<'PY'
import json
import sys
from pathlib import Path

run, build, request = map(Path, sys.argv[1:4])
step, seed, episodes = sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
if seed < 900 or episodes < 1:
    raise ValueError("Invalid held-out seed or episode count")
completed = json.loads((run / "completed.json").read_text())
if completed["trained_timesteps"] != 41_943_040:
    raise ValueError("Long run is incomplete")
build_config = json.loads((build / "build.json").read_text())["config"]
environment = build_config["python_environment"]
options = environment["options"]
if not (options["coworld_classic"] and options["parallel_games"] == 1024 and environment["spec"]["agents"] == 1024):
    raise ValueError("Build does not match the Classic four-buffer policy")
relative = completed["final_checkpoint"] if step == "final" else (
    f"checkpoints/metta_generals/run/{int(step):016d}.bin"
)
if relative not in completed["checkpoints"]:
    raise ValueError("Checkpoint is absent from the completed run")
checkpoint = run / relative
if not checkpoint.is_file():
    raise ValueError("Checkpoint file is missing")
request.mkdir(exist_ok=False)
(request / "config.json").write_text(json.dumps({
    "seeds": [seed], "episodes_per_seed": episodes, "timeout_seconds": 1200,
}) + "\n")
(request / "checkpoint.txt").write_text(str(checkpoint) + "\n")
PY

export run build output request
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-coworld
    printf "%s  %s\n" 2261a533cdbc89d4111ddf9d108c74252da1cf74ed66cf92513c00683645995f \
        "$workspace/source/generals/agents/harvester_agent.py" | sha256sum -c -
    name="relh-classic-buffer4-eval-${SLURM_JOB_ID}"
    cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
    trap cleanup EXIT
    trap "exit 143" TERM
    checkpoint=$(cat "$request/checkpoint.txt")
    timeout -k 30s 22m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource \
        relh-generals-b300:20260923b bash /work/source/integrations/eval_puffer_gpu.sh \
            "/work/${run#"$workspace/"}" "/work/${build#"$workspace/"}" \
            "/work/${output#"$workspace/"}" "/work/${request#"$workspace/"}/config.json" \
            "/work/${checkpoint#"$workspace/"}"
'

python3 - "$output" "$request" "$VALIDATION_SEED" "$EPISODES_PER_SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output, request, seed, episodes = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
evaluation = json.loads((output / "evaluation.json").read_text())
checkpoint = Path((request / "checkpoint.txt").read_text().strip())
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if evaluation["checkpoint_sha256"] != digest:
    raise ValueError("Evaluated checkpoint digest differs")
if evaluation["config"]["seeds"] != [seed] or evaluation["config"]["episodes_per_seed"] != episodes:
    raise ValueError("Evaluation seed or episode count differs")
results = evaluation["results"]
if len(results) != 1 or results[0]["seed"] != seed or results[0]["games"] < episodes:
    raise ValueError("Held-out games are incomplete")
print(json.dumps({"seed": seed, "episodes": episodes, "perf": results[0]["perf"],
                  "checkpoint_sha256": digest, "output": str(output)}, sort_keys=True))
PY

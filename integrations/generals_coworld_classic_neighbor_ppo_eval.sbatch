#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-neighbor-ppo-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:22:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/neighbor-ppo-eval-%j.log
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    echo "Usage: sbatch $0 TRAIN_JOB_ID OPPONENT SEED [CHECKPOINT_STEP]" >&2
    exit 2
fi
export TRAIN_JOB_ID=$1 NEIGHBOR_PPO_EVAL_OPPONENT=$2 NEIGHBOR_PPO_EVAL_SEED=$3 NEIGHBOR_PPO_EVAL_STEP=${4:-4194304}
case "$NEIGHBOR_PPO_EVAL_OPPONENT" in
    expander_harvester|sentinel) ;;
    *) echo "Unsupported opponent: $NEIGHBOR_PPO_EVAL_OPPONENT" >&2; exit 2 ;;
esac

srun --export=ALL --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
printf "%s  %s\n" 0ee098ae92a1ce4e699d076a3249b329727e16305f218227d1ecc09a58767131 \
    "$workspace/source-neighbor-threat/integrations/metta_puffer.py" | sha256sum -c -
train_job=$TRAIN_JOB_ID
opponent=$NEIGHBOR_PPO_EVAL_OPPONENT
seed=$NEIGHBOR_PPO_EVAL_SEED
checkpoint_step=$NEIGHBOR_PPO_EVAL_STEP
run=$workspace/neighbor-ppo-$train_job/run
source_build=$workspace/neighbor-ppo-build-$train_job
build=$source_build
output=$workspace/neighbor-ppo-eval-$train_job-$opponent-$seed-$SLURM_JOB_ID
config=$workspace/neighbor-ppo-eval-$train_job-$opponent-$seed-$SLURM_JOB_ID.json
name=relh-classic-neighbor-ppo-eval-$SLURM_JOB_ID
padded=$(printf "%016d" "$checkpoint_step")
checkpoint=$run/checkpoints/metta_generals/run/${padded}.bin
transfer=true
build=$workspace/neighbor-ppo-target-build-$SLURM_JOB_ID
python3 - "$source_build/build.json" "$output.build-config.json" "$opponent" <<"PY"
import json
import sys
from pathlib import Path
source, output = map(Path, sys.argv[1:3])
manifest = json.loads(source.read_text())
assert manifest["config"]["python_environment"]["options"]["opponent"] == "strong_mixed"
manifest["config"]["python_environment"]["options"]["opponent"] = sys.argv[3]
output.write_text(json.dumps(manifest["config"]) + "\n")
PY
python3 - "$run" "$checkpoint" "$config" "$seed" "$transfer" <<"PY"
import hashlib
import json
import sys
from pathlib import Path
run, checkpoint, config = map(Path, sys.argv[1:4])
seed, transfer = int(sys.argv[4]), sys.argv[5] == "true"
completed = json.loads((run / "completed.json").read_text())
assert completed["trained_timesteps"] == 4_194_304
assert checkpoint.is_file() and str(checkpoint.relative_to(run)) in completed["checkpoints"]
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "48bef910b750da0ca7e3068e921f5780cbb820199587e9ce5f0993b4764b8b0d"
config.write_text(json.dumps({"seeds": [seed], "episodes_per_seed": 1,
                              "allow_environment_transfer": transfer,
                              "timeout_seconds": 900}) + "\n")
print("checkpoint_sha256=" + hashlib.sha256(checkpoint.read_bytes()).hexdigest())
PY
docker_common=(--rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-neighbor-threat:/work/oldsource)
cleanup() { docker stop -t 5 "${name}-build" "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 3m docker run "${docker_common[@]}" --name "${name}-build" \
        relh-generals-b300:20260923b bash /work/source-neighbor-threat/integrations/run_puffer_gpu.sh build \
        --config "/work/${output#"$workspace/"}.build-config.json" \
        --output "/work/${build#"$workspace/"}"
timeout -k 30s 17m docker run "${docker_common[@]}" --name "$name" \
    relh-generals-b300:20260923b bash /work/source-neighbor-threat/integrations/eval_puffer_gpu.sh \
    "/work/${run#"$workspace/"}" "/work/${build#"$workspace/"}" \
    "/work/${output#"$workspace/"}" "/work/${config#"$workspace/"}" \
    "/work/${checkpoint#"$workspace/"}"
python3 - "$output" "$checkpoint" "$opponent" "$seed" "$transfer" <<"PY"
import hashlib
import json
import sys
from pathlib import Path
output, checkpoint = map(Path, sys.argv[1:3])
opponent, seed, transfer = sys.argv[3], int(sys.argv[4]), sys.argv[5] == "true"
record = json.loads((output / "evaluation.json").read_text())
assert record["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert (record["source_build"] is not None) == transfer
assert record["build"]["config"]["python_environment"]["options"]["opponent"] == opponent
assert record["config"]["seeds"] == [seed]
assert record["config"]["allow_environment_transfer"] == transfer
assert len(record["results"]) == 1 and record["results"][0]["games"] >= 1
print(json.dumps({"opponent": opponent, "seed": seed,
                  "result": record["results"][0]}, sort_keys=True))
PY
'

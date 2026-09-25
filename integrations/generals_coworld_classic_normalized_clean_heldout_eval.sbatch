#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-clean-heldout
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/normalized-clean-heldout-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-clean-heldout-$job
run=$w/normalized-clean-pilot-16788/run
checkpoint=$run/checkpoints/metta_generals/run/0000000004194304.bin
printf "%s  %s\n" 169ef0df7107a7bdc82d6836341cde62be507b3e4e189e1e977d631a219487a6 "$checkpoint" | sha256sum -c -
test -s "$run/completed.json"
printf "%s  %s\n" bd35a2359a62ee8160511f91f2244a92bb883a6ff5fa22802cb4d0a4ca0ca9c6 \
    "$w/metta_environment-main-fast-16245.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; break; fi
done
test -n "$gpu"
common=(--rm --gpus "device=$gpu" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work"
    -v "$w/metta_environment-main-fast-16245.py:/work/oldsource/metta_training/environment.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource)
cleanup() { docker stop -t 5 "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for opponent in random expander_harvester sentinel; do
    prefix=normalized-heldout-16734-$opponent
    build=$w/$prefix-build
    config=$w/$prefix-eval-config.json
    output=$w/normalized-clean-heldout-$job-$opponent-eval
    python3 - "$w/normalized-clean-build-16788/build.json" "$build/build.json" "$config" "$opponent" <<"PY"
import json, sys
from pathlib import Path
train, target = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
assert train["model_sha256"] == target["model_sha256"]
assert train["model_state_words"] == target["model_state_words"]
assert target["config"]["python_environment"]["options"]["opponent"] == sys.argv[4]
config = json.loads(Path(sys.argv[3]).read_text())
assert config["seeds"] == [1101] and config["allow_environment_transfer"]
PY
    echo "EVALUATE opponent=$opponent"
    timeout -k 30s 15m docker run "${common[@]}" --name "$name-eval" \
        relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/eval_puffer_gpu.sh \
        /work/normalized-clean-pilot-16788/run "/work/$prefix-build" \
        "/work/normalized-clean-heldout-$job-$opponent-eval" "/work/$prefix-eval-config.json" \
        /work/normalized-clean-pilot-16788/run/checkpoints/metta_generals/run/0000000004194304.bin \
        > "$w/normalized-clean-heldout-$job-$opponent-eval.log" 2>&1
    python3 - "$output/evaluation.json" "$opponent" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == "169ef0df7107a7bdc82d6836341cde62be507b3e4e189e1e977d631a219487a6"
assert record["build"]["config"]["python_environment"]["options"]["opponent"] == sys.argv[2]
assert record["config"]["seeds"] == [1101]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
done
'

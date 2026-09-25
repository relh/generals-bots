#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-normalized-heldout
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/normalized-heldout-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-normalized-heldout-$job
run=$w/normalized-zerocopy-pilot-16633/run
checkpoint=$run/checkpoints/metta_generals/run/0000000004194304.bin
printf "%s  %s\n" 2d3d1166cc5ac57c52fbada38de79ff50f3eb946bc3e91ad6856056d86818407 \
    "$checkpoint" | sha256sum -c -
printf "%s  %s\n" bd35a2359a62ee8160511f91f2244a92bb883a6ff5fa22802cb4d0a4ca0ca9c6 \
    "$w/metta_environment-main-fast-16245.py" | sha256sum -c -
test -s "$run/run.json"
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
cleanup() { docker stop -t 5 "$name-build" "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for opponent in random expander_harvester sentinel; do
    echo "EVALUATE opponent=$opponent"
    prefix=normalized-heldout-$job-$opponent
    build=$w/$prefix-build
    output=$w/$prefix-eval
    build_config=$w/$prefix-build-config.json
    eval_config=$w/$prefix-eval-config.json
    python3 - "$w/normalized-zerocopy-build-16633/build.json" "$build_config" "$eval_config" "$opponent" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
config = manifest["config"]
assert config["fabric"]["options"]["normalized_global"] is True
assert config["python_environment"]["options"]["coworld_classic"] is True
assert config["python_environment"]["options"]["opponent"] == "random"
config["python_environment"]["options"]["opponent"] = sys.argv[4]
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
Path(sys.argv[3]).write_text(json.dumps({
    "seeds": [1101], "episodes_per_seed": 1,
    "allow_environment_transfer": True, "timeout_seconds": 900,
}) + "\n")
PY
    timeout -k 30s 5m docker run "${common[@]}" --name "$name-build" \
        relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
        --config "/work/$prefix-build-config.json" --output "/work/$prefix-build" \
        > "$w/$prefix-build.log" 2>&1
    python3 - "$w/normalized-zerocopy-build-16633/build.json" "$build/build.json" "$opponent" <<"PY"
import json, sys
from pathlib import Path
original, target = (json.loads(Path(path).read_text()) for path in sys.argv[1:3])
assert original["model_sha256"] == target["model_sha256"]
assert original["model_state_words"] == target["model_state_words"]
assert original["environment_sha256"] != target["environment_sha256"]
assert target["config"]["python_environment"]["options"]["opponent"] == sys.argv[3]
PY
    timeout -k 30s 17m docker run "${common[@]}" --name "$name-eval" \
        relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/eval_puffer_gpu.sh \
        /work/normalized-zerocopy-pilot-16633/run "/work/$prefix-build" \
        "/work/$prefix-eval" "/work/$prefix-eval-config.json" \
        /work/normalized-zerocopy-pilot-16633/run/checkpoints/metta_generals/run/0000000004194304.bin \
        > "$w/$prefix-eval.log" 2>&1
    python3 - "$output/evaluation.json" "$opponent" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == "2d3d1166cc5ac57c52fbada38de79ff50f3eb946bc3e91ad6856056d86818407"
assert record["build"]["config"]["python_environment"]["options"]["opponent"] == sys.argv[2]
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
done
'

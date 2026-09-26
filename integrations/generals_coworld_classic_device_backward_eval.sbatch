#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-backward-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/device-backward-eval-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/device-backward-eval-$job
checkpoint=$w/device-backward-v3-pilot-18082/run/checkpoints/metta_generals/run/0000000004194304.bin
mkdir -p "$out"
printf "%s  %s\n" e726bedc53211f43afcfd8dc9e5f15f99e6b55d95eab4e4c7c0edd8b00309499 "$checkpoint" | sha256sum -c -
printf "%s  %s\n" a6473c7211257996408d4ecbc804600692960e3435633eee4da61cf58333f742 \
    "$w/metta-device-backward-source-v2.tar.gz" | sha256sum -c -
printf "%s  %s\n" 285f1782bed8450601fe59d5884710c193c3a4c8ceb807f9e00edfbc00fe00e8 \
    "$w/metta_puffer-hintreuse.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
python3 - "$w/device-backward-v2-build-18076/build.json" "$out/build-config.json" \
    "$w/normalized-heldout-16734-expander_harvester-eval-config.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["model_sha256"] == "a8fdf4cef1cbdb157206b237959234d71c37fb0a36ff4ee5289464f66d7333f4"
config = manifest["config"]
config["environment_backend"] = "cpu"
config["python_environment"]["device_resident"] = False
config["python_environment"]["spec"]["agents"] = 1024
config["python_environment"]["options"]["parallel_games"] = 1024
config["python_environment"]["options"]["opponent"] = "expander_harvester"
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
evaluation = json.loads(Path(sys.argv[3]).read_text())
assert evaluation["seeds"] == [1101] and evaluation["allow_environment_transfer"]
Path(sys.argv[2]).with_name("eval-config.json").write_text(json.dumps(evaluation) + "\n")
PY
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
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v "$w/metta_puffer-hintreuse.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache
    -e PYTHONPATH=/work/metta-device-backward-source-v2:/work/runtime:/work/source-normalized-16245:/work/oldsource)
name=relh-classic-device-backward-eval-$job
cleanup() { docker stop -t 5 "$name-build" "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/device-backward-eval-$job/build-config.json" \
    --output "/work/device-backward-eval-$job/build" > "$out/build-container.log" 2>&1
python3 - "$w/device-backward-v2-build-18076/build.json" "$out/build/build.json" <<"PY"
import json, sys
from pathlib import Path
train, target = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
assert train["model_sha256"] == target["model_sha256"]
assert train["model_state_words"] == target["model_state_words"]
assert target["config"]["environment_backend"] == "cpu"
assert target["config"]["python_environment"]["spec"]["teacher"]
assert target["config"]["python_environment"]["options"]["opponent"] == "expander_harvester"
PY
timeout -k 30s 18m docker run "${common[@]}" --name "$name-eval" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/eval_puffer_gpu.sh \
    /work/device-backward-v3-pilot-18082/run "/work/device-backward-eval-$job/build" \
    "/work/device-backward-eval-$job/eval" "/work/device-backward-eval-$job/eval-config.json" \
    /work/device-backward-v3-pilot-18082/run/checkpoints/metta_generals/run/0000000004194304.bin \
    > "$out/eval-container.log" 2>&1
python3 - "$out/eval/evaluation.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == "e726bedc53211f43afcfd8dc9e5f15f99e6b55d95eab4e4c7c0edd8b00309499"
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
'

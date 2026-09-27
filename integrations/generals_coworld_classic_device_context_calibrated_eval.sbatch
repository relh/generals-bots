#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-calibrated-eval
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
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-calibrated-eval-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-calibrated-eval-$job
checkpoint=$disk/device-context-calibrated-mb131072-${CALIBRATED_TRAIN_JOB:?}/run/checkpoints/metta_generals/run/0000000016777216.bin
mkdir -p "$out"
printf "%s  %s\n" fbf8e4a12d96da6bf2fb32e63b10916b7d3699c360b5d33ad2b1a3493a9c3509 \
    "$disk/launch_puffer_calibrated_transfer.py" | sha256sum -c -
printf "%s  %s\n" 9f59b9047fa642c8dd4815f268d60cadba67fb691480ee82dd9d68f6de806626 \
    "$disk/run_puffer_calibrated_transfer.sh" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
python3 - "$checkpoint" "$disk/device-context-calibrated-mb131072-${CALIBRATED_TRAIN_JOB:?}/run/completed.json" <<"PY"
import hashlib, json, sys
from pathlib import Path
checkpoint, result = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
assert checkpoint.name == Path(result["final_checkpoint"]).name
print("EVAL_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest(), flush=True)
PY
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" eb034d4dcf79dd4c4a16d65595c3195cba676c06b008003af34c7cedef469502 \
    "$disk/metta_puffer-calibrated-v1.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
python3 - "$disk/device-context-calibrated-16384-build-${CALIBRATED_BUILD_JOB:?}/build/build.json" "$out/build-config.json" \
    "$w/normalized-heldout-16734-expander_harvester-eval-config.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["fabric"]["teacher"]["phases"][0]["coefficient"] == 0.0
assert manifest["config"]["python_environment"]["options"]["move_hint_scale"] == .375
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
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
common=(--rm --gpus "device=$gpu" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v "$disk/metta_puffer-calibrated-v1.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource)
name=relh-classic-device-context-calibrated-eval-$job
cleanup() { docker stop -t 5 "$name-build" "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 8m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/recovery/device-context-calibrated-eval-$job/build-config.json" \
    --output "/recovery/device-context-calibrated-eval-$job/build" > "$out/build-container.log" 2>&1
timeout -k 30s 18m docker run "${common[@]}" --name "$name-eval" \
    relh-generals-b300:20260923b bash /recovery/run_puffer_calibrated_transfer.sh evaluate \
    --run /recovery/device-context-calibrated-mb131072-${CALIBRATED_TRAIN_JOB:?}/run \
    --build "/recovery/device-context-calibrated-eval-$job/build" \
    --output "/recovery/device-context-calibrated-eval-$job/eval" \
    --config "/recovery/device-context-calibrated-eval-$job/eval-config.json" \
    --checkpoint /recovery/device-context-calibrated-mb131072-${CALIBRATED_TRAIN_JOB:?}/run/checkpoints/metta_generals/run/0000000016777216.bin \
    > "$out/eval-container.log" 2>&1
python3 - "$out/eval/evaluation.json" "$checkpoint" <<"PY"
import hashlib, json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
'

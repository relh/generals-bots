#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-v11-transfer-eval
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
#SBATCH --output=/tmp/relh-generals-coworld/device-v11-transfer-eval-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/device-v11-transfer-eval-$job
checkpoint=$w/device-v11-transfer-pilot-17779/run/checkpoints/metta_generals/run/0000000004194304.bin
mkdir -p "$out"
printf "%s  %s\n" fe5b25260619567c3ce8127783da3a951b5e8255f1a16a002545a1ec42f3d2ac "$checkpoint" | sha256sum -c -
printf "%s  %s\n" 8dc2689bea60e46b0089ec613f69877daefd813f15236460159d7990febcbaef \
    "$w/metta-v11-transfer-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" 03b3527ac714a88596739da2d56747eecbdb1405c6c2f80f954c6dcc26ea7d85 \
    "$w/metta_puffer-device-hinted.py" | sha256sum -c -
printf "%s  %s\n" cacf6a57180f176f9ac7d77384cd9b1f9b33250f0240888ff63284778d496706 \
    "$w/puffer_codec-device-current.py" | sha256sum -c -
python3 - "$w/device-v11-transfer-build-17773/build.json" "$out/build-config.json" \
    "$w/normalized-heldout-16734-expander_harvester-eval-config.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["model_sha256"] == "5ba4d2d8d83cf6f7ae3276269d9f39519a502857d6a730577f7a80dbd81e247f"
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
    -v "$w/metta_puffer-device-hinted.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$w/puffer_codec-device-current.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache
    -e PYTHONPATH=/work/metta-v11-transfer-source:/work/runtime:/work/source-normalized-16245:/work/oldsource)
name=relh-classic-device-v11-transfer-eval-$job
cleanup() { docker stop -t 5 "$name-build" "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/device-v11-transfer-eval-$job/build-config.json" \
    --output "/work/device-v11-transfer-eval-$job/build" > "$out/build-container.log" 2>&1
python3 - "$w/device-v11-transfer-build-17773/build.json" "$out/build/build.json" <<"PY"
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
    /work/device-v11-transfer-pilot-17779/run "/work/device-v11-transfer-eval-$job/build" \
    "/work/device-v11-transfer-eval-$job/eval" "/work/device-v11-transfer-eval-$job/eval-config.json" \
    /work/device-v11-transfer-pilot-17779/run/checkpoints/metta_generals/run/0000000004194304.bin \
    > "$out/eval-container.log" 2>&1
python3 - "$out/eval/evaluation.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == "fe5b25260619567c3ce8127783da3a951b5e8255f1a16a002545a1ec42f3d2ac"
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-overnight67m-sentinel-eval
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
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-overnight67m-sentinel-eval-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-overnight67m-sentinel-eval-$job
checkpoint=$disk/device-context-overnight-22922/run/checkpoints/metta_generals/run/0000000067108864.bin
mkdir -p "$out"
printf "%s  %s\n" 8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f "$checkpoint" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" ea5106c8467f10465d39e6a2779cca58fee464f791e1f7f36665041a76357151 \
    "$w/metta_puffer-device-fused.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
python3 - "$w/device-context-build-18326/build.json" "$out/build-config.json" \
    "$w/normalized-heldout-16734-expander_harvester-eval-config.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["model_sha256"] == "84e2952b84980bfdf29ca287f63031d85b77d29c9909546b4ab2bfee291dbd96"
config = manifest["config"]
config["environment_backend"] = "cpu"
config["python_environment"]["device_resident"] = False
config["python_environment"]["spec"]["agents"] = 1024
config["python_environment"]["options"]["parallel_games"] = 1024
config["python_environment"]["options"]["opponent"] = "sentinel"
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
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v "$w/metta_puffer-device-fused.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource)
name=relh-classic-device-context-overnight67m-sentinel-eval-$job
cleanup() { docker stop -t 5 "$name-build" "$name-eval" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/recovery/device-context-overnight67m-sentinel-eval-$job/build-config.json" \
    --output "/recovery/device-context-overnight67m-sentinel-eval-$job/build" > "$out/build-container.log" 2>&1
python3 - "$out/build/build.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["model_sha256"] == "84e2952b84980bfdf29ca287f63031d85b77d29c9909546b4ab2bfee291dbd96"
assert record["model_state_words"] == 22956
assert record["config"]["python_environment"]["options"]["opponent"] == "sentinel"
PY
timeout -k 30s 18m docker run "${common[@]}" --name "$name-eval" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/eval_puffer_gpu.sh \
    /recovery/device-context-overnight-22922/run "/recovery/device-context-overnight67m-sentinel-eval-$job/build" \
    "/recovery/device-context-overnight67m-sentinel-eval-$job/eval" "/recovery/device-context-overnight67m-sentinel-eval-$job/eval-config.json" \
    /recovery/device-context-overnight-22922/run/checkpoints/metta_generals/run/0000000067108864.bin \
    > "$out/eval-container.log" 2>&1
python3 - "$out/eval/evaluation.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record["checkpoint_sha256"] == "8ee76d545e459dc6eaf37e952c38f2c14f6083968541f06a4ce0e2cdc69d993f"
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-supervised-build
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/device-supervised-build-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
printf "%s  %s\n" 23636a3753f3b120922d746df20ece5db7ed91b1f0eb9d89070146b09122ba60 \
    "$w/metta-device-supervised-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" 4c38b397bed9920442bf477d1004f7990691ed695900022beb68db3f9df2ced0 \
    "$w/metta_puffer-device-supervised.py" | sha256sum -c -
python3 - "$w/normalized-clean-build-16788/build.json" "$w/device-supervised-build-$job-config.json" <<"PY"
import json, sys
from pathlib import Path
old = json.loads(Path(sys.argv[1]).read_text())
config = old["config"]
assert config["python_environment"]["options"]["coworld_classic"] is True
assert config["fabric"]["options"]["normalized_global"] is True
config["environment_backend"] = "cuda"
config["python_environment"]["device_resident"] = True
config["python_environment"]["spec"]["agents"] = 4096
config["python_environment"]["options"]["parallel_games"] = 4096
config["python_environment"]["options"].update({
    "opponent": "strong_mixed", "teacher": "expander_harvester",
    "imitation_weight": 0.0, "supervise_teacher": True,
})
config["python_environment"]["spec"]["teacher"] = True
config["fabric"]["teacher"] = {"phases": [{"agent_steps": 0, "ppo_coefficient": 0, "coefficient": 1, "action_mix": 1}]}
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
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
name=relh-classic-device-supervised-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-supervised.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-supervised-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/device-supervised-build-$job-config.json" \
    --output "/work/device-supervised-build-$job" > "$w/device-supervised-build-$job-container.log" 2>&1
python3 - "$w/normalized-clean-build-16788/build.json" "$w/device-supervised-build-$job/build.json" <<"PY"
import json, sys
from pathlib import Path
old, new = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
fabric = new["config"]["fabric"].copy()
assert fabric.pop("train_mask_column") is None
teacher = fabric.pop("teacher")
assert teacher["phases"][0]["action_mix"] == 1
baseline_fabric = old["config"]["fabric"].copy()
assert baseline_fabric.pop("teacher") is None
assert fabric == baseline_fabric
assert new["model_state_words"] == old["model_state_words"]
assert new["config"]["environment_backend"] == "cuda"
assert new["config"]["python_environment"]["device_resident"]
assert new["config"]["python_environment"]["options"]["opponent"] == "strong_mixed"
assert new["config"]["python_environment"]["options"]["supervise_teacher"]
assert new["config"]["python_environment"]["spec"]["teacher"]
print("DEVICE_BRIDGE_BUILD_OK", new["binary_sha256"], flush=True)
PY
'

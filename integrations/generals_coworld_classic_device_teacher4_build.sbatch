#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-teacher4-build
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
#SBATCH --output=/tmp/relh-generals-coworld/device-teacher4-build-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
printf "%s  %s\n" 33e408f05a68dd3e2b9b3d5f2d8375009cd3141ebb346d50d9ece63d0194d30d \
    "$w/metta-device-bridge-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e4e69fdfc88602c950e50547e1c0b6d5f8b16c2f35c7d93342e478c07fe3183f \
    "$w/metta_puffer-device-bridge.py" | sha256sum -c -
python3 - "$w/normalized-clean-build-16788/build.json" "$w/device-teacher4-build-$job-config.json" <<"PY"
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
    "opponent": "strong_mixed", "teacher": "expander_harvester", "imitation_weight": 4.0,
})
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
name=relh-classic-device-teacher4-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-bridge.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-bridge-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/device-teacher4-build-$job-config.json" \
    --output "/work/device-teacher4-build-$job" > "$w/device-teacher4-build-$job-container.log" 2>&1
python3 - "$w/normalized-clean-build-16788/build.json" "$w/device-teacher4-build-$job/build.json" <<"PY"
import json, sys
from pathlib import Path
old, new = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
fabric = new["config"]["fabric"].copy()
assert fabric.pop("train_mask_column") is None
assert fabric == old["config"]["fabric"]
assert new["model_state_words"] == old["model_state_words"]
assert new["config"]["environment_backend"] == "cuda"
assert new["config"]["python_environment"]["device_resident"]
assert new["config"]["python_environment"]["options"]["opponent"] == "strong_mixed"
assert new["config"]["python_environment"]["options"]["imitation_weight"] == 4.0
print("DEVICE_BRIDGE_BUILD_OK", new["binary_sha256"], flush=True)
PY
'

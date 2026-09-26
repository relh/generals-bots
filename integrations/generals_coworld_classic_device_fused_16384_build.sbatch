#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-fused-16384-build
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
#SBATCH --output=/tmp/relh-generals-coworld/device-fused-16384-build-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
printf "%s  %s\n" a6473c7211257996408d4ecbc804600692960e3435633eee4da61cf58333f742 \
    "$w/metta-device-backward-source-v2.tar.gz" | sha256sum -c -
printf "%s  %s\n" ea5106c8467f10465d39e6a2779cca58fee464f791e1f7f36665041a76357151 \
    "$w/metta_puffer-device-fused.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
python3 - "$w/source-normalized-16245/integrations/configs/gpu-coworld-classic-expander-prior-build.json" "$w/device-fused-16384-build-$job-config.json" <<"PY"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
options = config["python_environment"]["options"]
assert options["hint_features"] and options["prior_hint_features"] and options["expander_hint_features"]
assert options["teacher"] == "expander_harvester"
config.update({"mode": "train", "environment_backend": "cuda", "precision": "float32"})
config["python_environment"].update({"workers": "inline", "device_resident": True})
config["python_environment"]["spec"].update({"agents": 16384, "teacher": True, "replay_metadata_size": 0})
options.update({
    "parallel_games": 16384, "opponent": "strong_mixed", "sparse_teacher": False,
    "supervise_teacher": True, "imitation_weight": 0.0,
})
fabric = config["fabric"]
fabric["replay_metadata_size"] = 0
fabric["losses"] = []
fabric["teacher"] = {"phases": [{"agent_steps": 0, "ppo_coefficient": 1, "coefficient": 0.25, "action_mix": 0}]}
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
name=relh-classic-device-fused-16384-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-fused.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-backward-source-v2:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/device-fused-16384-build-$job-config.json" \
    --output "/work/device-fused-16384-build-$job" > "$w/device-fused-16384-build-$job-container.log" 2>&1
python3 - "$w/source-normalized-16245/integrations/configs/gpu-coworld-classic-expander-prior-build.json" "$w/device-fused-16384-build-$job/build.json" <<"PY"
import json, sys
from pathlib import Path
reference, new = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
assert reference["python_environment"]["options"]["expander_hint_features"]
assert new["config"]["environment_backend"] == "cuda"
assert new["config"]["python_environment"]["device_resident"]
assert new["config"]["python_environment"]["spec"]["teacher"]
assert new["config"]["python_environment"]["spec"]["observation_size"] == 3528
assert new["config"]["fabric"]["options"]["hint_prior_strength"] == 8.0
phase = new["config"]["fabric"]["teacher"]["phases"][0]
assert (phase["ppo_coefficient"], phase["coefficient"], phase["action_mix"]) == (1.0, 0.25, 0.0)
assert new["model_state_words"] > 0
print("DEVICE_HINTED_BUILD_OK", new["binary_sha256"], new["model_sha256"], flush=True)
PY
'

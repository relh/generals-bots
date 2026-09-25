#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-normalized-clean
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/normalized-clean-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/normalized-clean-pilot-$job
build=$w/normalized-clean-build-$job
name=relh-classic-normalized-clean-$job
mkdir -p "$out"
printf "%s  %s\n" bd35a2359a62ee8160511f91f2244a92bb883a6ff5fa22802cb4d0a4ca0ca9c6 \
    "$w/metta_environment-main-fast-16245.py" | sha256sum -c -
printf "%s  %s\n" 9a3db919d671740bc76a0aa90d3a74b0ae9a5120132ffc93006a6c772ce73690 \
    "$w/monitor_coworld_steady_interval_600.py" | sha256sum -c -
python3 - "$w/normalized-build-16245/build.json" "$out/config.json" "$out/build-config.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["fabric"]["options"]["normalized_global"] is True
assert manifest["config"]["python_environment"]["options"]["coworld_classic"] is True
assert manifest["config"]["python_environment"]["options"]["opponent"] == "random"
assert manifest["config"]["python_environment"]["options"]["teacher"] is None
record = {
    "total_timesteps": 4_194_304,
    "seed": 711,
    "overrides": {
        "vec.total_agents": 4096, "vec.num_threads": 4, "vec.num_buffers": 4,
        "train.horizon": 32, "train.minibatch_size": 16384,
        "train.replay_ratio": 0.25, "train.learning_rate": 0.001,
        "train.max_grad_norm": 0.5, "train.ent_coef": 0.0001,
        "train.vf_coef": 1.0, "train.gamma": 0.999,
        "base.eval_episodes": 0, "base.checkpoint_interval": 32,
        "sweep.downsample": 1,
    },
}
Path(sys.argv[2]).write_text(json.dumps(record) + "\n")
Path(sys.argv[3]).write_text(json.dumps(manifest["config"]) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
gpu_local=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; gpu_local=$i; break; fi
done
test -n "$gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$gpu local_index=$gpu_local"
common=(--rm --gpus "device=$gpu" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work"
    -v "$w/metta_environment-main-fast-16245.py:/work/oldsource/metta_training/environment.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-normalized-16245:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" "$name-build" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 5m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/normalized-clean-pilot-$job/build-config.json" \
    --output "/work/normalized-clean-build-$job" > "$out/build.log" 2>&1
python3 - "$w/normalized-build-16245/build.json" "$build/build.json" <<"PY"
import json, sys
from pathlib import Path
original, changed = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
assert original["model_sha256"] == changed["model_sha256"]
assert original["model_state_words"] == changed["model_state_words"]
assert original["environment_sha256"] != changed["environment_sha256"]
print("normalized_model_sha256=" + changed["model_sha256"])
print("fast_transport_environment_sha256=" + changed["environment_sha256"])
PY
ln -s "$out/run" "$w/normalized-clean-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$w/normalized-clean-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 16m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh train \
    --config "/work/normalized-clean-pilot-$job/config.json" \
    --build "/work/normalized-clean-build-$job" \
    --output "/work/normalized-clean-pilot-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval_600.py" \
        "$w" "$job" "$((now-start))" normalized-clean 600; then
        echo "Normalized clean PPO pilot stopped by 30K SPS or 600-second startup guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

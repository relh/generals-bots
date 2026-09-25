#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-castle-ppo-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/castle-ppo-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/castle-ppo-$job
build=$workspace/castle-ppo-build-$job
name=relh-classic-castle-ppo-$job
mkdir -p "$output"
printf "%s  %s\n" 88f2bb78bd8ac089fb905f411b8b22cf9e5501ae8b8ff5594365fd888dda25b6 \
    "$workspace/source-policy-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" 415870756502faef3f7bfd4caa34a6f2cd8b1ebcc50f7d1b3db1f1093dc49617 \
    "$workspace/source-castle-ppo/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" b4ae076013ea3e4d35512ff04c87ed0308bc2966759220f75f5e80e3a094af3a \
    "$workspace/source-castle-ppo/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" a744a03f00ebd0a8fc75518a733bfab275d729baa81c5df0d3bb845332fb64af \
    "$workspace/ppo-strong-build-13486/build.json" | sha256sum -c -
printf "%s  %s\n" 939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499 \
    "$workspace/buffer4-long-13316/run/checkpoints/metta_generals/run/0000000031457280.bin" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
if (( ${#allocated_gpus[@]} != ${#usage[@]} )); then
    echo "Slurm allocation and visible GPU list differ" >&2
    exit 1
fi
chosen_gpu=
gpu_local=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then
        chosen_gpu=${allocated_gpus[$i]}
        gpu_local=$i
        break
    fi
done
if [[ -z $chosen_gpu ]]; then
    echo "No uncontended B300 in Slurm allocation $SLURM_STEP_GPUS" >&2
    exit 1
fi
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$chosen_gpu local_index=$gpu_local"
python3 - "$workspace/ppo-strong-build-13486/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
assert manifest["model_state_words"] == 12360
options = manifest["config"]["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["opponent"] == "expander_harvester"
options["shaping_weight"] = 1.0
options["castle_shaping_weight"] = 1.0
assert manifest["config"]["fabric"]["losses"][0]["replace_ppo"] is False
build_config.write_text(json.dumps(manifest["config"]) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 629,
    "initialize": {
        "run": "/work/buffer4-long-13316/run",
        "checkpoint": "/work/buffer4-long-13316/run/checkpoints/metta_generals/run/0000000031457280.bin",
        "sha256": "939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499",
        "allow_environment_transfer": True,
    },
    "overrides": {
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 8192,
        "train.horizon": 32,
        "train.replay_ratio": 0.125,
        "train.learning_rate": 0.0001,
        "train.max_grad_norm": 0.5,
        "train.ent_coef": 0.005,
        "train.vf_coef": 1.0,
        "train.gamma": 0.99,
        "base.eval_episodes": 0,
        "sweep.downsample": 1,
        "base.checkpoint_interval": 16,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work
    -e PYTHONPATH=/work/runtime:/work/source-policy-transfer:/work/source-castle-ppo:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-castle-ppo && python -m pytest tests/test_metta_puffer.py -q -k castle_control_margin" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-castle-ppo/integrations/run_puffer_gpu.sh build \
    --config "/work/castle-ppo-$job/build-config.json" \
    --output "/work/castle-ppo-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["model_sha256"] == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
assert manifest["model_state_words"] == 12360
PY
ln -s "$output/run" "$workspace/castle-ppo-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/castle-ppo-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-castle-ppo/integrations/run_puffer_gpu.sh train \
    --config "/work/castle-ppo-$job/config.json" \
    --build "/work/castle-ppo-build-$job" \
    --output "/work/castle-ppo-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-castle-ppo/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" castle-ppo 1; then
        echo "Castle PPO pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

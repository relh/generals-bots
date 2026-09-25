#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-distance-ppo-pilot
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
#SBATCH --output=/tmp/relh-generals-coworld/distance-ppo-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/distance-ppo-$job
build=$workspace/distance-ppo-build-$job
name=relh-classic-distance-ppo-$job
mkdir -p "$output"
printf "%s  %s\n" e9d68296fc076b9a9d4fce7b6951a5eae19279836dcc3e99ab0209280a01983d \
    "$workspace/source-packed-ppo-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" f3582f20dede5ccdfb84edb5652ce2a652dc1cd3ca942d23d35adb6c7ae3e2bd \
    "$workspace/source-general-distance/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 574cf9066664d61cae1595d3367e917d8b811bb4f67ed26fb10c16dbe0b0e687 \
    "$workspace/source-general-distance/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 22dffab046bf6eec78ba822f87792865008993b783b31c87575ff74556b1802e \
    "$workspace/distance-ppo-prebuild-15106/build/build.json" | sha256sum -c -
printf "%s  %s\n" bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31 \
    "$workspace/packed-hint-long-14603/run/checkpoints/metta_generals/run/0000000029360128.bin" | sha256sum -c -
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
python3 - "$workspace/distance-ppo-prebuild-15106/build/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43"
assert manifest["model_state_words"] == 14124
options = manifest["config"]["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["opponent"] == "strong_mixed"
assert options["general_distance_hint_features"] is True
assert options["packed_context_hint_features"] is False
assert options["shaping_weight"] == 1.0 and options["castle_shaping_weight"] == 1.0
assert manifest["config"]["fabric"]["losses"][0]["replace_ppo"] is False
assert manifest["config"]["fabric"]["losses"][0]["options"]["coefficient"] == 0.25
build_config.write_text(json.dumps(manifest["config"]) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 661,
    "initialize": {
        "run": "/work/packed-hint-long-14603/run",
        "checkpoint": "/work/packed-hint-long-14603/run/checkpoints/metta_generals/run/0000000029360128.bin",
        "sha256": "bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31",
        "allow_environment_transfer": True,
    },
    "overrides": {
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 16384,
        "train.horizon": 32,
        "train.replay_ratio": 0.125,
        "train.learning_rate": 0.001,
        "train.max_grad_norm": 0.5,
        "train.ent_coef": 0.005,
        "train.vf_coef": 1.0,
        "train.gamma": 0.99,
        "base.eval_episodes": 0,
        "sweep.downsample": 1,
        "base.checkpoint_interval": 32,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work
    -e PYTHONPATH=/work/runtime:/work/source-packed-ppo-transfer:/work/source-general-distance:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-general-distance && python -m pytest tests/test_metta_puffer.py -q -k general_distance_hints && python -m pytest tests/test_coworld_neural_codec.py -q -k general_distance" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh build \
    --config "/work/distance-ppo-$job/build-config.json" \
    --output "/work/distance-ppo-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["model_sha256"] == "40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43"
assert manifest["model_state_words"] == 14124
PY
ln -s "$output/run" "$workspace/distance-ppo-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/distance-ppo-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh train \
    --config "/work/distance-ppo-$job/config.json" \
    --build "/work/distance-ppo-build-$job" \
    --output "/work/distance-ppo-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-general-distance/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" distance-ppo 1; then
        echo "General-distance PPO pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

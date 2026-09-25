#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-packed-ppo-mb16-pilot
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
#SBATCH --output=/tmp/relh-generals-coworld/packed-ppo-mb16-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/packed-ppo-mb16-$job
build=$workspace/packed-ppo-mb16-build-$job
name=relh-classic-packed-ppo-mb16-$job
mkdir -p "$output"
printf "%s  %s\n" e9d68296fc076b9a9d4fce7b6951a5eae19279836dcc3e99ab0209280a01983d \
    "$workspace/source-packed-ppo-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" 6a8e261770153df827a8b6260a374e943f41e4d2f077ae7e71c3a21c534b8d96 \
    "$workspace/source-packed-sentinel/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 2a947eacd631ee207f9e888ba1e298dd7aa482f05ed1832dcc47fc976af30e84 \
    "$workspace/source-packed-sentinel/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 0483144365784d1daa39dd1560fffc6434e98ecf888111c7ce201118da688a27 \
    "$workspace/packed-ppo-prebuild-14883/build/build.json" | sha256sum -c -
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
python3 - "$workspace/packed-ppo-prebuild-14883/build/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
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
assert options["shaping_weight"] == 1.0 and options["castle_shaping_weight"] == 1.0
assert manifest["config"]["fabric"]["losses"][0]["replace_ppo"] is False
assert manifest["config"]["fabric"]["losses"][0]["options"]["coefficient"] == 0.25
build_config.write_text(json.dumps(manifest["config"]) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 659,
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
        "base.checkpoint_interval": 16,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work
    -e PYTHONPATH=/work/runtime:/work/source-packed-ppo-transfer:/work/source-packed-sentinel:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-packed-sentinel && python -m pytest tests/test_metta_puffer.py -q -k strong_mixed_opponents && python -m pytest tests/test_metta_puffer.py -q -k packed_context_sentinel_only" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-packed-sentinel/integrations/run_puffer_gpu.sh build \
    --config "/work/packed-ppo-mb16-$job/build-config.json" \
    --output "/work/packed-ppo-mb16-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["model_sha256"] == "40cbc1d5e997b15fbd32f4732dc0fc6a1e7f346709c709a987556a57cdb65c43"
assert manifest["model_state_words"] == 14124
PY
ln -s "$output/run" "$workspace/packed-ppo-mb16-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/packed-ppo-mb16-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-packed-sentinel/integrations/run_puffer_gpu.sh train \
    --config "/work/packed-ppo-mb16-$job/config.json" \
    --build "/work/packed-ppo-mb16-build-$job" \
    --output "/work/packed-ppo-mb16-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-packed-sentinel/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" packed-ppo-mb16 1; then
        echo "Packed PPO minibatch-16K pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-cardinal2-ppo-pilot
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
#SBATCH --output=/tmp/relh-generals-coworld/cardinal2-ppo-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/cardinal2-ppo-$job
build=$workspace/cardinal2-ppo-build-$job
name=relh-classic-cardinal2-ppo-$job
mkdir -p "$output"
printf "%s  %s\n" 23cc9aaefe6539a65411682093c30d0c78fdff613c33b769c318bb0ed93d077d \
    "$workspace/source-cardinal2-ppo-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" f3582f20dede5ccdfb84edb5652ce2a652dc1cd3ca942d23d35adb6c7ae3e2bd \
    "$workspace/source-general-distance/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 574cf9066664d61cae1595d3367e917d8b811bb4f67ed26fb10c16dbe0b0e687 \
    "$workspace/source-general-distance/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" dc1067e66f92c4f7c38567beee93e7197bc8c83e99f34dad8936ac18b1711106 \
    "$workspace/cardinal2-ppo-prebuild-15226/build/build.json" | sha256sum -c -
printf "%s  %s\n" fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0 \
    "$workspace/cardinal2-mb16-long-15207/run/checkpoints/metta_generals/run/0000000025165824.bin" | sha256sum -c -
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
python3 - "$workspace/cardinal2-ppo-prebuild-15226/build/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "ba1e2e49188a4ff66191bc9091ea80832119ccd09d542c8ddfd217fb5776eabf"
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
    "seed": 673,
    "initialize": {
        "run": "/work/cardinal2-mb16-long-15207/run",
        "checkpoint": "/work/cardinal2-mb16-long-15207/run/checkpoints/metta_generals/run/0000000025165824.bin",
        "sha256": "fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0",
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
    -e PYTHONPATH=/work/runtime:/work/source-cardinal2-ppo-transfer:/work/source-general-distance:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-general-distance && python -m pytest tests/test_metta_puffer.py -q -k strong_mixed_opponents && python -m pytest tests/test_metta_puffer.py -q -k packed_context_hints" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh build \
    --config "/work/cardinal2-ppo-$job/build-config.json" \
    --output "/work/cardinal2-ppo-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["model_sha256"] == "ba1e2e49188a4ff66191bc9091ea80832119ccd09d542c8ddfd217fb5776eabf"
assert manifest["model_state_words"] == 14124
PY
ln -s "$output/run" "$workspace/cardinal2-ppo-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/cardinal2-ppo-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh train \
    --config "/work/cardinal2-ppo-$job/config.json" \
    --build "/work/cardinal2-ppo-build-$job" \
    --output "/work/cardinal2-ppo-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-general-distance/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" cardinal2-ppo 1; then
        echo "Cardinal PPO pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

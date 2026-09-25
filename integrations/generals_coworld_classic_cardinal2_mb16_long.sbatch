#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-cardinal2-mb16-long-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/cardinal2-mb16-long-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/cardinal2-mb16-long-$job
build=$workspace/cardinal2-mb16-long-build-$job
name=relh-classic-cardinal2-mb16-long-$job
mkdir -p "$output"
printf "%s  %s\n" f3582f20dede5ccdfb84edb5652ce2a652dc1cd3ca942d23d35adb6c7ae3e2bd \
    "$workspace/source-general-distance/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 574cf9066664d61cae1595d3367e917d8b811bb4f67ed26fb10c16dbe0b0e687 \
    "$workspace/source-general-distance/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" ce893e5cc45a81df06921dc638c18be4beadf70faec4f38177a8c286787cd9c0 \
    "$workspace/cardinal2-mb16-build-15189/build.json" | sha256sum -c -
printf "%s  %s\n" cacf6a57180f176f9ac7d77384cd9b1f9b33250f0240888ff63284778d496706 \
    "$workspace/source-general-distance/integrations/puffer_codec.py" | sha256sum -c -
printf "%s  %s\n" 90b903f95bee7a43f92cbe1cfe300db9cf9e01802b96a70bdee8acebcc15bb15 \
    "$workspace/source-general-distance/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" 5af84bc27504c08e787de9f33e0c4ad0ad450a2049b1fcd50313c6d5b41b48cf \
    "$workspace/source-general-distance/tests/test_coworld_neural_codec.py" | sha256sum -c -
printf "%s  %s\n" cc420fd829e3408405a772029296881972ba72ff64b8d56e329825eb8307f5c4 \
    "$workspace/cardinal2-mb16-15189/run/checkpoints/metta_generals/run/0000000004194304.bin" | sha256sum -c -
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
python3 - "$workspace/cardinal2-mb16-build-15189/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5"
assert manifest["model_state_words"] == 14124
options = manifest["config"]["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["opponent"] == "expander_harvester"
assert options["packed_context_hint_features"] is True
assert manifest["config"]["fabric"]["observation_size"] == 10 * 21 * 21
assert manifest["config"]["fabric"]["losses"][0]["replace_ppo"] is True
build_config.write_text(json.dumps(manifest["config"]) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 25_165_824,
    "seed": 671,
    "initialize": {
        "run": "/work/cardinal2-mb16-15189/run",
        "checkpoint": "/work/cardinal2-mb16-15189/run/checkpoints/metta_generals/run/0000000004194304.bin",
        "sha256": "cc420fd829e3408405a772029296881972ba72ff64b8d56e329825eb8307f5c4",
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
        "train.ent_coef": 0.0,
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
    -e PYTHONPATH=/work/runtime:/work/source-general-distance:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-general-distance && python -m pytest tests/test_metta_puffer.py -q -k packed_context_hints && python -m pytest tests/test_coworld_neural_codec.py -q -k wire_view" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh build \
    --config "/work/cardinal2-mb16-long-$job/build-config.json" \
    --output "/work/cardinal2-mb16-long-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["config"]["fabric"]["observation_size"] == 10 * 21 * 21
assert manifest["config"]["fabric"]["options"]["channels"] == 10
assert manifest["model_sha256"] == "ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5"
assert manifest["model_state_words"] == 14124
PY
ln -s "$output/run" "$workspace/cardinal2-mb16-long-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/cardinal2-mb16-long-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 29m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-general-distance/integrations/run_puffer_gpu.sh train \
    --config "/work/cardinal2-mb16-long-$job/config.json" \
    --build "/work/cardinal2-mb16-long-build-$job" \
    --output "/work/cardinal2-mb16-long-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-general-distance/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" cardinal2-mb16-long 1; then
        echo "Cardinal mb16 long run stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

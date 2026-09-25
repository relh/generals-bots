#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-v11-castle-ppo-exact
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
#SBATCH --output=/tmp/relh-generals-coworld/v11-castle-ppo-exact-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/v11-castle-ppo-$job
build=$workspace/v11-castle-ppo-build-$job
name=relh-classic-v11-castle-ppo-$job
mkdir -p "$output"
printf "%s  %s\n" 4e6a5d2abaa9d9c2bc31db15b1da4d55caa169e6ad2ce96c596de478afa74c55 \
    "$workspace/source-v11-ppo-transfer/metta_training/puffer.py" | sha256sum -c -
printf "%s  %s\n" f932ec2fa90f29af4aa9314f37728bd886319fc15bede046807a40f90741892b \
    "$workspace/source-v11-castle/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 27152b741e8338c2d145977c26fcdaa57f857e35e23ec0e3d64894123b9bbb00 \
    "$workspace/source-v11-castle/tests/test_metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 574cf9066664d61cae1595d3367e917d8b811bb4f67ed26fb10c16dbe0b0e687 \
    "$workspace/source-v11-castle/tests/test_metta_puffer_castle.py" | sha256sum -c -
printf "%s  %s\n" 7ef182b2e978fadd3cbfe2792251ef9b50823d1c24b51c9141c36aa65ec95bdf \
    "$workspace/monitor_coworld_steady_interval.py" | sha256sum -c -
printf "%s  %s\n" ca9190625d4321e31efd5310f699a64cad1e623273e1c819b9ba2ff91aa5202c \
    "$workspace/v11-ppo-build-15292/build/build.json" | sha256sum -c -
printf "%s  %s\n" 8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346 \
    "$workspace/hybrid-clean-13916/run/checkpoints/metta_generals/run/0000000033554432.bin" | sha256sum -c -
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
python3 - "$workspace/v11-ppo-build-15292/build/build.json" "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
assert manifest["model_state_words"] == 12360
options = manifest["config"]["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["opponent"] == "strong_mixed"
assert options["sentinel_teacher_fraction"] == 1.0 and options["sentinel_teacher_interval"] == 4
assert options["shaping_weight"] == 1.0 and options["castle_shaping_weight"] == 1.0
assert manifest["config"]["fabric"]["losses"][0]["replace_ppo"] is False
assert manifest["config"]["fabric"]["losses"][0]["options"]["coefficient"] == 0.25
build_config.write_text(json.dumps(manifest["config"]) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 677,
    "initialize": {
        "run": "/work/hybrid-clean-13916/run",
        "checkpoint": "/work/hybrid-clean-13916/run/checkpoints/metta_generals/run/0000000033554432.bin",
        "sha256": "8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346",
        "allow_environment_transfer": True,
    },
    "overrides": {
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 16384,
        "train.horizon": 32,
        "train.replay_ratio": 0.125,
        "train.learning_rate": 0.0003,
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
    -e PYTHONPATH=/work/runtime:/work/source-v11-ppo-transfer:/work/source-v11-castle:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-test" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${common[@]}" --name "$name-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-v11-castle && python -m pytest tests/test_metta_puffer.py -q -k sentinel_teacher && python -m pytest tests/test_metta_puffer_castle.py -q -k castle_control_margin" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-v11-castle/integrations/run_puffer_gpu.sh build \
    --config "/work/v11-castle-ppo-$job/build-config.json" \
    --output "/work/v11-castle-ppo-build-$job" > "$output/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json
import sys
manifest = json.load(open(sys.argv[1]))
assert manifest["model_sha256"] == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
assert manifest["model_state_words"] == 12360
PY
ln -s "$output/run" "$workspace/v11-castle-ppo-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/v11-castle-ppo-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-v11-castle/integrations/run_puffer_gpu.sh train \
    --config "/work/v11-castle-ppo-$job/config.json" \
    --build "/work/v11-castle-ppo-build-$job" \
    --output "/work/v11-castle-ppo-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/monitor_coworld_steady_interval.py" \
        "$workspace" "$job" "$((now-start))" v11-castle-ppo; then
        echo "v11 castle PPO pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

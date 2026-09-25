#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-sentinel-only-pilot
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
#SBATCH --output=/tmp/relh-generals-coworld/sentinel-only-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:3 --cpus-per-task=16 bash -lc '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-sentinel-only-$job
output=$workspace/sentinel-only-$job
build=$workspace/sentinel-only-build-$job
mkdir -p "$output"
printf "%s  %s\n" 2261a533cdbc89d4111ddf9d108c74252da1cf74ed66cf92513c00683645995f \
    "$workspace/source-sentinel-only/generals/agents/harvester_agent.py" | sha256sum -c -
printf "%s  %s\n" 35aed77fa70d0a68c237f84d183a50dc59fb238018ce9e57cccb756e56a0c6a3 \
    "$workspace/source-sentinel-only/integrations/monitor_coworld_long.py" | sha256sum -c -
printf "%s  %s\n" efc34d0bf7fa8fede48dde048a241c9950197cee3ebc788fed5aab1607f53366 \
    "$workspace/source-sentinel-only/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" 7f1e71b2286d90b6a990bcfba0bf7aa87339ec04b95e3a7f52e55e5483f6d71b \
    "$workspace/source-sentinel-only/tests/test_metta_puffer.py" | sha256sum -c -
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
python3 - "$workspace/buffer4-long-13316/run/run.json" \
    "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
config = manifest["build"]["config"]
environment = config["python_environment"]
options = environment["options"]
if not (options["coworld_classic"] and options["parallel_games"] == 1024
        and environment["spec"]["agents"] == 1024):
    raise ValueError("Expected 1,024-game Coworld Classic build")
options["opponent"] = "expander_harvester"
options["sentinel_teacher_fraction"] = 1.0
options["sentinel_teacher_interval"] = 4
options["sentinel_teacher_only"] = True
losses = config["fabric"]["losses"]
if len(losses) != 1 or losses[0]["name"] != "sparse_teacher" or not losses[0]["replace_ppo"]:
    raise ValueError("Expected the sparse-teacher-only training baseline")

build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 627,
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
        "train.learning_rate": 0.0003,
        "train.max_grad_norm": 0.5,
        "train.ent_coef": 0.0,
        "train.vf_coef": 1.0,
        "train.gamma": 0.99,
        "base.eval_episodes": 0,
        "sweep.downsample": 1,
        "base.checkpoint_interval": 16,
    },
}) + "\n")
PY
docker_common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-sentinel-only:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "${name}-test" "${name}-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 2m docker run "${docker_common[@]}" --name "${name}-test" \
    relh-generals-b300:20260923b bash -lc \
    "cd /work/source-sentinel-only && python -m pytest tests/test_metta_puffer.py -q -k sentinel" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run "${docker_common[@]}" --name "${name}-build" \
    relh-generals-b300:20260923b bash /work/source-sentinel-only/integrations/run_puffer_gpu.sh build \
    --config "/work/sentinel-only-$job/build-config.json" \
    --output "/work/sentinel-only-build-$job" > "$output/build.log" 2>&1
ln -s "$output/run" "$workspace/sentinel-only-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/sentinel-only-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${docker_common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-sentinel-only/integrations/run_puffer_gpu.sh train \
    --config "/work/sentinel-only-$job/config.json" \
    --build "/work/sentinel-only-build-$job" \
    --output "/work/sentinel-only-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-sentinel-only/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" sentinel-only 1; then
        echo "Sentinel-only pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

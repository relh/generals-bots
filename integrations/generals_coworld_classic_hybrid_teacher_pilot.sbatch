#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-hybrid-teacher-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/hybrid-teacher-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -lc '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-hybrid-teacher-$job
output=$workspace/hybrid-teacher-$job
build=$workspace/hybrid-teacher-build-$job
mkdir -p "$output"
printf "%s  %s\n" 2261a533cdbc89d4111ddf9d108c74252da1cf74ed66cf92513c00683645995f \
    "$workspace/source-hybrid-teacher/generals/agents/harvester_agent.py" | sha256sum -c -
printf "%s  %s\n" 35aed77fa70d0a68c237f84d183a50dc59fb238018ce9e57cccb756e56a0c6a3 \
    "$workspace/source-hybrid-teacher/integrations/monitor_coworld_long.py" | sha256sum -c -
printf "%s  %s\n" 0f201425d51865368ed8a9ee7d02bbb4d95599b360e918e890baebfc04df2f8d \
    "$workspace/source-hybrid-teacher/integrations/metta_puffer.py" | sha256sum -c -
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
options["sentinel_teacher_fraction"] = 0.5
losses = config["fabric"]["losses"]
if len(losses) != 1 or losses[0]["name"] != "sparse_teacher" or not losses[0]["replace_ppo"]:
    raise ValueError("Expected the sparse-teacher-only training baseline")

build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 621,
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
docker_common=(--rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-hybrid-teacher:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "${name}-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 6m docker run "${docker_common[@]}" --name "${name}-build" \
    relh-generals-b300:20260923b bash /work/source-hybrid-teacher/integrations/run_puffer_gpu.sh build \
    --config "/work/hybrid-teacher-$job/build-config.json" \
    --output "/work/hybrid-teacher-build-$job" > "$output/build.log" 2>&1
ln -s "$output/run" "$workspace/hybrid-teacher-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/hybrid-teacher-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${docker_common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-hybrid-teacher/integrations/run_puffer_gpu.sh train \
    --config "/work/hybrid-teacher-$job/config.json" \
    --build "/work/hybrid-teacher-build-$job" \
    --output "/work/hybrid-teacher-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source-hybrid-teacher/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" hybrid-teacher 1; then
        echo "Hybrid teacher pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-sentinel-compact
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/sentinel-compact-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -lc '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
name=relh-classic-sentinel-compact-$job
output=$workspace/sentinel-compact-$job
build=$workspace/sentinel-compact-build-$job
mkdir -p "$output"
printf "%s  %s\n" 2261a533cdbc89d4111ddf9d108c74252da1cf74ed66cf92513c00683645995f \
    "$workspace/source/generals/agents/harvester_agent.py" | sha256sum -c -
printf "%s  %s\n" 35aed77fa70d0a68c237f84d183a50dc59fb238018ce9e57cccb756e56a0c6a3 \
    "$workspace/source/integrations/monitor_coworld_long.py" | sha256sum -c -
python3 - "$workspace/expander-quad-stability-build-13219/build.json" \
    "$output/build-config.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
config = manifest["config"]
environment = config["python_environment"]
options = environment["options"]
if not (options["coworld_classic"] and options["parallel_games"] == 1024
        and environment["spec"]["agents"] == 1024):
    raise ValueError("Expected 1,024-game Coworld Classic build")
options["opponent"] = "expander_harvester"
options["teacher"] = "sentinel"
options["teacher_rollouts"] = True
options["lean_features"] = False
options["hint_features"] = False
options["prior_hint_features"] = False
options["expander_hint_features"] = False
environment["spec"]["observation_size"] = 14 * 21 * 21
config["fabric"]["observation_size"] = 14 * 21 * 21
config["fabric"]["options"].update({
    "channels": 14,
    "features_per_site": 2,
    "global_features": 2,
    "input_radius": 0.1,
    "hint_prior_strength": 0.0,
})
losses = config["fabric"]["losses"]
if len(losses) != 1 or losses[0]["name"] != "sparse_teacher" or not losses[0]["replace_ppo"]:
    raise ValueError("Expected the sparse-teacher-only training baseline")
losses[0]["options"]["coefficient"] = 1.0
build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 4_194_304,
    "seed": 613,
    "overrides": {
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 8192,
        "train.horizon": 32,
        "train.replay_ratio": 0.125,
        "train.learning_rate": 0.001,
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
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "${name}-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 3m docker run "${docker_common[@]}" --name "${name}-build" \
    relh-generals-b300:20260923b bash /work/source/integrations/run_puffer_gpu.sh build \
    --config "/work/sentinel-compact-$job/build-config.json" \
    --output "/work/sentinel-compact-build-$job" > "$output/build.log" 2>&1
ln -s "$output/run" "$workspace/sentinel-compact-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$workspace/sentinel-compact-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 10m docker run "${docker_common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source/integrations/run_puffer_gpu.sh train \
    --config "/work/sentinel-compact-$job/config.json" \
    --build "/work/sentinel-compact-build-$job" \
    --output "/work/sentinel-compact-$job/run" > "$output/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$workspace/source/integrations/monitor_coworld_long.py" \
        "$workspace" "$job" "$((now-start))" sentinel-compact 1; then
        echo "Sentinel compact pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$output/run/completed.json" || status=1
exit "$status"
'

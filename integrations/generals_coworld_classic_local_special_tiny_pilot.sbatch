#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-local-special-tiny
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/local-special-tiny-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/local-special-tiny-$job
build=$w/local-special-tiny-build-$job
name=relh-classic-local-special-tiny-$job
mkdir -p "$out"
printf "%s  %s\n" dc53d7646b0d4fa0648e1601fee2ebff6657581964dc08a841e2b674474403a8 \
    "$w/source-local-special-16245/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" 6aa6f23f6db3ae1eb3213c95c5820a8f4dc79035c2b963c408f643c35de980c4 \
    "$w/source-local-special-16245/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" a53522774e8474c3f3bf5107547043d3c32b16f91d23ccca9a0b0423ce75c67b \
    "$w/local-special-build-16245/build.json" | sha256sum -c -
printf "%s  %s\n" 7ef182b2e978fadd3cbfe2792251ef9b50823d1c24b51c9141c36aa65ec95bdf \
    "$w/monitor_coworld_steady_interval.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
(( ${#allocated_gpus[@]} == ${#usage[@]} ))
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
test -n "$chosen_gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$chosen_gpu local_index=$gpu_local"
python3 - "$w/local-special-build-16245/build.json" "$out/build-config.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
config = manifest["config"]
fabric = config["fabric"]
options = config["python_environment"]["options"]
assert fabric["factory"] == "integrations.generals_fabric:tied_local_action_policy"
assert fabric["options"]["local_special_readout"] is True
assert fabric["losses"] == [] and options["teacher"] is None
assert options["coworld_classic"] and not options["coworld_tiny_map_curriculum"]
options["coworld_tiny_map_curriculum"] = True
build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 8_388_608,
    "seed": 699,
    "overrides": {
        "vec.total_agents": 4096, "vec.num_threads": 4, "vec.num_buffers": 4,
        "train.minibatch_size": 16384, "train.horizon": 32,
        "train.replay_ratio": 0.375, "train.learning_rate": 0.001,
        "train.max_grad_norm": 0.5, "train.ent_coef": 0.0001,
        "train.vf_coef": 1.0, "train.gamma": 0.999,
        "base.eval_episodes": 0, "sweep.downsample": 1,
        "base.checkpoint_interval": 32,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-local-special-16245:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-preflight" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 5m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-local-special-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/local-special-tiny-$job/build-config.json" \
    --output "/work/local-special-tiny-build-$job" > "$out/build.log" 2>&1
python3 - "$w/local-special-build-16245/build.json" "$build/build.json" <<"PY"
import json, sys
parent, child = (json.load(open(path)) for path in sys.argv[1:])
assert child["model_sha256"] == parent["model_sha256"]
assert child["config"]["python_environment"]["options"]["coworld_tiny_map_curriculum"]
print("model_sha256=" + child["model_sha256"])
PY
timeout -k 30s 2m docker run -i "${common[@]}" --name "$name-preflight" \
    relh-generals-b300:20260923b python3 - "/work/local-special-tiny-$job/build-config.json" <<"PY" > "$out/preflight.log" 2>&1
import json, sys
from pathlib import Path
from metta_training.environment import EnvironmentContext
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
config = json.loads(Path(sys.argv[1]).read_text())
options = dict(config["python_environment"]["options"])
options["parallel_games"] = 1
env = BatchedGeneralsPufferEnvironment(
    context=EnvironmentContext(seed=699, index=0, mode="train", output=Path("/work")), **options,
)
assert env.spec.observation_size == 14 * 21 * 21
assert env.base.env.truncation == 300 and env.base.env.min_grid_size == 6
assert options["opponent"] == "random" and options["teacher"] is None
print("local_special_tiny_preflight=ok", flush=True)
PY
ln -s "$out/run" "$w/local-special-tiny-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$w/local-special-tiny-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 15m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-local-special-16245/integrations/run_puffer_gpu.sh train \
    --config "/work/local-special-tiny-$job/config.json" \
    --build "/work/local-special-tiny-build-$job" \
    --output "/work/local-special-tiny-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval.py" \
        "$w" "$job" "$((now-start))" local-special-tiny; then
        echo "Local-special tiny PPO pilot stopped by the 30K SPS or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

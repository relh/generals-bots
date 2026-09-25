#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-raw-ppo-tuned-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/raw-ppo-tuned-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/raw-ppo-tuned-$job
build=$w/raw-ppo-tuned-build-$job
name=relh-classic-raw-ppo-tuned-$job
mkdir -p "$out"
printf "%s  %s\n" 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 \
    "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" 459e10eab59671c160d11ae8f49a3ffa31e487adf3a606ac7fedc3748745a56d \
    "$w/source-spatial2/integrations/metta_puffer.py" | sha256sum -c -
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e \
    "$w/hybrid-periodic-build-13829/build.json" | sha256sum -c -
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
python3 - "$w/hybrid-periodic-build-13829/build.json" "$out/build-config.json" "$out/config.json" <<"PY"
import json
import sys
from pathlib import Path
source, build_config, train_config = map(Path, sys.argv[1:])
manifest = json.loads(source.read_text())
assert manifest["model_sha256"] == "c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9"
config = manifest["config"]
fabric = config["fabric"]
environment = config["python_environment"]
options = environment["options"]
assert fabric["losses"][0]["replace_ppo"] is True
assert fabric["losses"][0]["options"]["coefficient"] == 1.0
assert options["coworld_classic"] and options["sentinel_teacher_interval"] == 4
assert options["opponent"] == "expander_harvester"
options.update({
    "opponent": "mixed", "shaping_weight": 1.0, "teacher": None, "sparse_teacher": False,
    "supervise_teacher": False, "imitation_weight": 0.0,
    "compact_features": False, "lean_features": False,
    "hint_features": False, "prior_hint_features": False,
    "expander_hint_features": False,
    "sentinel_teacher_fraction": 0.0, "sentinel_teacher_interval": 1,
})
environment["spec"]["observation_size"] = 14 * 21 * 21
environment["spec"]["replay_metadata_size"] = 0
fabric["observation_size"] = 14 * 21 * 21
fabric["replay_metadata_size"] = 0
fabric["losses"] = []
fabric["factory"] = "integrations.generals_fabric:two_stage_tied_local_action_policy"
fabric["options"] = {
    "channels": 14, "height": 21, "width": 21,
    "features_per_site": 2, "global_features": 2,
    "context_radius": 1.01, "hint_prior_strength": 0.0,
}
build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 8_388_608,
    "seed": 691,
    "overrides": {
        "vec.total_agents": 4096, "vec.num_threads": 4, "vec.num_buffers": 4,
        "train.minibatch_size": 16384, "train.horizon": 32,
        "train.replay_ratio": 0.5, "train.learning_rate": 0.0003,
        "train.max_grad_norm": 0.5, "train.ent_coef": 0.0001,
        "train.vf_coef": 1.0, "train.gamma": 0.99,
        "base.eval_episodes": 0, "sweep.downsample": 1,
        "base.checkpoint_interval": 32,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-spatial2:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-preflight" "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 7m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/work/raw-ppo-tuned-$job/build-config.json" \
    --output "/work/raw-ppo-tuned-build-$job" > "$out/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["config"]["fabric"]["factory"].endswith(":two_stage_tied_local_action_policy")
assert d["config"]["fabric"]["losses"] == []
assert d["config"]["python_environment"]["options"]["teacher"] is None
assert d["config"]["fabric"]["observation_size"] == 14 * 21 * 21
assert d["model_state_words"] > 0
print("model_sha256=" + d["model_sha256"])
print("model_state_words=" + str(d["model_state_words"]))
PY
timeout -k 30s 2m docker run "${common[@]}" --name "$name-preflight" \
    relh-generals-b300:20260923b python3 - "/work/raw-ppo-tuned-$job/build-config.json" <<"PY" > "$out/preflight.log" 2>&1
import json
import sys
from pathlib import Path
from metta_training.environment import EnvironmentContext
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
config = json.loads(Path(sys.argv[1]).read_text())
options = dict(config["python_environment"]["options"])
options["parallel_games"] = 1
env = BatchedGeneralsPufferEnvironment(
    context=EnvironmentContext(seed=691, index=0, mode="train", output=Path("/work")),
    **options,
)
assert env.spec.observation_size == 14 * 21 * 21
assert env.spec.replay_metadata_size == 0 and not env.spec.teacher
assert env.base.prior_hint_features is False and env.base.sparse_teacher is False
print("raw_observation_ppo_preflight=ok", flush=True)
PY
ln -s "$out/run" "$w/raw-ppo-tuned-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$w/raw-ppo-tuned-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 17m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh train \
    --config "/work/raw-ppo-tuned-$job/config.json" \
    --build "/work/raw-ppo-tuned-build-$job" \
    --output "/work/raw-ppo-tuned-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval.py" \
        "$w" "$job" "$((now-start))" raw-ppo-tuned; then
        echo "Raw-observation PPO pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

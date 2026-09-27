#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-teacher-h128
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-teacher-h128-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/classic-selfplay-teacher-h128-$job
build=$disk/classic-selfplay-teacher-h128-build-$job
name=relh-classic-selfplay-teacher-h128-$job
mkdir -p "$out"
printf "%s  %s\n" 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 \
    "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" a0261c5d81806b8ba2b4ed9675dda0dc51bd08c29109c6ef92dc3cb26c470886 \
    "$disk/metta_puffer-selfplay-teacher-v2-20260927.py" | sha256sum -c -
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e \
    "$w/hybrid-periodic-build-13829/build.json" | sha256sum -c -
printf "%s  %s\n" 9b88eb675c9781f4f1a4b670577cf61ce490622a89d22f16321f3d0213f028e0 \
    "$disk/monitor_coworld_steady_interval-v2.py" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
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
config.update({"mode": "train", "environment_backend": "cuda", "precision": "float32"})
environment.update({"workers": "inline", "device_resident": True})
environment["spec"]["agents"] = 8192
options.update({
    "opponent": "strong_mixed", "teacher": "expander_harvester", "sparse_teacher": False,
    "supervise_teacher": True, "imitation_weight": 0.0, "land_gain_reward_weight": 0.2,
    "shaping_weight": 1.0, "shaping_gamma": 0.999, "reward_scale": 0.5,
    "army_shaping_weight": 0.5, "land_shaping_weight": 0.3, "castle_shaping_weight": 0.0,
    "balance_opponent_sides": True, "deduplicate_opponent_branches": True,
    "parallel_games": 4096,
    "compact_features": True, "lean_features": True,
    "hint_features": True, "prior_hint_features": True,
    "expander_hint_features": True, "context_hint_features": True,
    "sentinel_teacher_fraction": 0.0, "sentinel_teacher_interval": 1,
})
environment["spec"]["observation_size"] = 14 * 21 * 21
environment["spec"]["replay_metadata_size"] = 0
environment["spec"]["teacher"] = True
environment["factory"] = "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
fabric["observation_size"] = 14 * 21 * 21
fabric["replay_metadata_size"] = 0
fabric["losses"] = []
fabric["teacher"] = {"interpolation": "constant", "phases": [
    {"agent_steps": 0, "ppo_coefficient": 0.0, "coefficient": 1.0, "action_mix": 1.0},
    {"agent_steps": 16_777_216, "ppo_coefficient": 1.0, "coefficient": 0.01, "action_mix": 0.0},
]}
fabric["compiler"] = "standard"
fabric["factory"] = "integrations.generals_fabric:two_stage_tied_local_action_policy"
fabric["options"] = {
    "channels": 14, "height": 21, "width": 21,
    "features_per_site": 4, "global_features": 8,
    "context_radius": 1.01, "hint_prior_strength": 0.5,
}
build_config.write_text(json.dumps(config) + "\n")
train_config.write_text(json.dumps({
    "total_timesteps": 33_554_432,
    "seed": 714,
    "overrides": {
        "vec.total_agents": 8192, "vec.num_threads": 1, "vec.num_buffers": 1,
        "train.minibatch_size": 32768, "train.horizon": 128,
        "train.replay_ratio": 0.5, "train.learning_rate": 0.001,
        "train.max_grad_norm": 0.5, "train.ent_coef": 0.005, "train.gae_lambda": 0.99,
        "train.vf_coef": 1.0, "train.gamma": 0.999,
        "base.eval_episodes": 0, "sweep.downsample": 1,
        "base.checkpoint_interval": 4, "base.cudagraphs": -1,
    },
}) + "\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -v "$disk/metta_puffer-selfplay-teacher-v2-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v /var/tmp/relh-generals-recovery/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name-build" "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 7m docker run "${common[@]}" --name "$name-build" \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/recovery/classic-selfplay-teacher-h128-$job/build-config.json" \
    --output "/recovery/classic-selfplay-teacher-h128-build-$job" > "$out/build.log" 2>&1
python3 - "$build/build.json" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
assert d["config"]["fabric"]["factory"].endswith(":two_stage_tied_local_action_policy")
assert d["config"]["environment_backend"] == "cuda"
assert d["config"]["python_environment"]["device_resident"] is True
assert d["config"]["python_environment"]["spec"]["agents"] == 8192
assert d["config"]["python_environment"]["spec"]["teacher"] is True
assert d["config"]["python_environment"]["factory"].endswith(":BatchedGeneralsSelfPlayPufferEnvironment")
assert d["config"]["python_environment"]["options"]["parallel_games"] == 4096
assert d["config"]["python_environment"]["options"]["shaping_gamma"] == 0.999
assert d["config"]["python_environment"]["options"]["supervise_teacher"] is True
assert d["config"]["fabric"]["losses"] == []
assert d["config"]["fabric"]["compiler"] == "standard"
assert [p["action_mix"] for p in d["config"]["fabric"]["teacher"]["phases"]] == [1.0, 0.0]
assert d["config"]["python_environment"]["options"]["teacher"] == "expander_harvester"
assert d["config"]["fabric"]["observation_size"] == 14 * 21 * 21
assert d["model_state_words"] > 0
print("model_sha256=" + d["model_sha256"])
print("model_state_words=" + str(d["model_state_words"]))
PY
ln -s "$out/run" "$disk/classic-selfplay-teacher-h128-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/classic-selfplay-teacher-h128-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 22m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh train \
    --config "/recovery/classic-selfplay-teacher-h128-$job/config.json" \
    --build "/recovery/classic-selfplay-teacher-h128-build-$job" \
    --output "/recovery/classic-selfplay-teacher-h128-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$job" "$((now-start))" classic-selfplay-teacher-h128 600 1048576 30000; then
        echo "Self-play pilot stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

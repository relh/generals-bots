#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-calibrated-h128
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-calibrated-h128-%j.log
set -euo pipefail

# Mandatory user policy: verify lowest scheduling priority before workload startup.
slurm_task_priority_record=$(scontrol show job -o "${SLURM_JOB_ID:?}")
[[ " $slurm_task_priority_record " == *" Nice=2147483645 "* ]] || { printf 'Lowest Slurm Nice verification failed\n' >&2; exit 1; }
slurm_task_priority=unknown
for slurm_task_field in $slurm_task_priority_record; do
  case "$slurm_task_field" in Priority=*) slurm_task_priority=${slurm_task_field#Priority=} ;; esac
done
[[ "$slurm_task_priority" =~ ^[0-9]+$ ]] || exit 1
printf 'SLURM_PRIORITY_READBACK JobId=%s Nice=2147483645 Priority=%s\n' "$SLURM_JOB_ID" "$slurm_task_priority" >&2
unset slurm_task_priority_record slurm_task_field


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/classic-selfplay-iter1-calibrated-h128-$job
build=$disk/classic-selfplay-iter1-build-28048/build
name=relh-classic-selfplay-iter1-calibrated-h128-$job
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
parent=$disk/classic-selfplay-init134-h128-27957/run
printf "%s  %s\n" e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf \
    "$parent/checkpoints/metta_generals/run/0000000033554432.bin" | sha256sum -c -
printf "%s  %s\n" 05b27c5dc486d45baf86bc4f31ac99d5e246ffaa3bf418feff43259a83896e1d \
    "$build/build.json" | sha256sum -c -
printf "%s  %s\n" e56cd5a8834af5231441404cd51366259d65a5009e0c000e7682df0eecda084f \
    "$disk/puffer_selfplay_iter1_transfer-v1-20260927.py" | sha256sum -c -
printf "%s  %s\n" 3426b5c5a5e236f399871ed9795dfdf348f05564c011ff0d7e11841c0610bf92 \
    "$disk/launch_classic_selfplay_iter1_training-v1-20260927.py" | sha256sum -c -
python3 - "$disk/classic-selfplay-init134-h128-27957/config.json" "$out/config.json" "$build/build.json" <<"PYCONFIG"
import json,sys
from pathlib import Path
source,output,build=map(Path,sys.argv[1:])
config=json.loads(source.read_text())
manifest=json.loads(build.read_text())
assert config["total_timesteps"] == 33_554_432
assert config["seed"] == 715 and config["initialize"]["sha256"] == "c6cc6db63b3b5855d1ab7f38b99baf8ed25cfcb51f2c34819a43a1d58b51fe3a"
assert config["overrides"]["vec.total_agents"] == 8192
assert config["overrides"]["train.horizon"] == 128
assert config["overrides"]["train.minibatch_size"] == 32768
assert config["overrides"]["train.gamma"] == manifest["config"]["python_environment"]["options"]["shaping_gamma"] == 0.999
assert manifest["model_sha256"] == "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
assert manifest["model_state_words"] == 22956
config["seed"] = 716
config["initialize"] = {
    "run": "/recovery/classic-selfplay-init134-h128-27957/run",
    "checkpoint": "/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin",
    "sha256": "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf",
    "allow_environment_transfer": True,
    "allow_policy_only_transfer": True,
}
config["overrides"]["train.learning_rate"] = 0.0003
config["overrides"]["train.ent_coef"] = 0.003
config["overrides"]["train.anneal_lr"] = 0
output.write_text(json.dumps(config)+"\n")
print("initialized_from="+config["initialize"]["sha256"])
PYCONFIG
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -v "$disk/metta_puffer-selfplay-teacher-v2-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v /var/tmp/relh-generals-recovery/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
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
assert [p["action_mix"] for p in d["config"]["fabric"]["teacher"]["phases"]] == [0.0]
assert d["config"]["fabric"]["teacher"]["phases"][0]["coefficient"] == 0.0
assert d["config"]["python_environment"]["options"]["move_hint_scale"] == 0.25
assert d["config"]["python_environment"]["options"]["split_hint_scale"] == 0.25
assert d["config"]["python_environment"]["options"]["teacher"] == "expander_harvester"
assert d["config"]["fabric"]["observation_size"] == 14 * 21 * 21
assert d["model_state_words"] == 22956
assert d["model_sha256"] == "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
print("model_sha256=" + d["model_sha256"])
print("model_state_words=" + str(d["model_state_words"]))
PY
ln -s "$out/run" "$disk/classic-selfplay-iter1-calibrated-h128-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/classic-selfplay-iter1-calibrated-h128-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 22m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b python /recovery/launch_classic_selfplay_iter1_training-v1-20260927.py train \
    --config "/recovery/classic-selfplay-iter1-calibrated-h128-$job/config.json" \
    --build "/recovery/classic-selfplay-iter1-build-28048/build" \
    --output "/recovery/classic-selfplay-iter1-calibrated-h128-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$job" "$((now-start))" classic-selfplay-iter1-calibrated-h128 600 1048576 30000; then
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

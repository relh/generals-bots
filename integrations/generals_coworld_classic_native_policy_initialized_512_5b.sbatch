#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-initialized-512-5b
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:30:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-initialized-512-5b-%j.log
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
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-initialized-512-5b-$job
mkdir -p "$out"
printf "%s  %s\n" 1608c7a8a4450790c8fbf13940e7229cfc9c917a288db26686080b458ff42623 \
    "$disk/launch_native_initialized_training-20260927.py" | sha256sum -c -
printf "%s  %s\n" 9b88eb675c9781f4f1a4b670577cf61ce490622a89d22f16321f3d0213f028e0 \
    "$disk/monitor_coworld_steady_interval_300k-20260927.py" | sha256sum -c -
printf "%s  %s\n" 0958d85285144647aea4d32fc60f08e3965d82d4cad83f7b184a58949cd63ec4 \
    "$disk/puffer-native-initializer-20260927.py" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e75f1862f88847c8893344c8ff26ca0bb35ff284d085ff95b1b000f1c7833d75 \
    "$disk/metta_puffer-balanced-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
python3 - "$disk/device-context-native-policy-65536-build-25526/build/build.json" <<"PY"
import hashlib, json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
assert hashlib.sha256(manifest_path.with_name("puffer").read_bytes()).hexdigest() == manifest["binary_sha256"]
assert manifest["model_state_words"] == 0
assert manifest["model_sha256"] == ""
config = manifest["config"]
assert config["python_environment"]["spec"]["agents"] == 65536
assert not config["python_environment"]["spec"]["teacher"]
options = config["python_environment"]["options"]
assert options["opponent"] == "strong_mixed"
assert options["teacher"] == "expander_harvester"
assert not options["supervise_teacher"] and options["imitation_weight"] == 0.0
assert options["expander_hint_features"] and options["prior_hint_features"]
assert config["fabric"] is None
assert options["move_hint_scale"] == .375 and options["split_hint_scale"] == .125
assert options["shaping_gamma"] == .999
assert options["balance_opponent_sides"]
PY
python3 - "$w/normalized-clean-pilot-16788/config.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
record["total_timesteps"] = 5_368_709_120
record["seed"] = 1350
record["native_policy_initializer"] = None
record["initialize"] = {
    "run": "/recovery/native-policy-initialized-512-probe-25880/run",
    "checkpoint": "/recovery/native-policy-initialized-512-probe-25880/run/checkpoints/metta_generals/run/0000000033554432.bin",
    "sha256": "61dfb5f6bb06ec1e409f4ae8250d51872be7ff5004900a48deb84b12e52b1a08",
}
assert record["overrides"]["train.gamma"] == .999
record["overrides"].update({
    "vec.total_agents": 65536, "vec.num_buffers": 1, "vec.num_threads": 1,
    "base.cudagraphs": -1, "base.checkpoint_interval": 128,
    "policy.hidden_size": 512, "policy.num_layers": 1,
    "train.learning_rate": 0.003, "train.ent_coef": 0.0, "train.replay_ratio": 1.0, "train.horizon": 16, "train.minibatch_size": 524288,
})
Path(sys.argv[2]).write_text(json.dumps(record) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$gpu"
name=relh-classic-native-policy-initialized-512-5b-$job
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-policy-initialized-512-5b-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-policy-initialized-512-5b-pilot-$job-0"
timeout -k 30s 315m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib -e PYTHONFAULTHANDLER=1 -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b python /recovery/launch_native_initialized_training-20260927.py train \
    --config /recovery-output/config.json \
    --build /recovery/device-context-native-policy-65536-build-25526/build \
    --output /recovery-output/run > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_300k-20260927.py" \
        "$disk" "$job" "$((now-start))" native-policy-initialized-512-5b 600 1048576 300000; then
        echo "Device bridge smoke stopped by startup or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
if (( status == 0 )); then
python3 - "$out/run" <<"PY"
import hashlib, json, math, struct, sys
from pathlib import Path
run = Path(sys.argv[1])
completed = json.loads((run / "completed.json").read_text())
assert completed["trained_timesteps"] == 5_368_709_120
checkpoint = run / completed["final_checkpoint"]
data = checkpoint.read_bytes()
assert len(data) > 0 and len(data) % 4 == 0
assert all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data))
print("NATIVE_PARAMETERS_FINITE", len(data) // 4, hashlib.sha256(data).hexdigest(), flush=True)
PY
fi
exit "$status"
'

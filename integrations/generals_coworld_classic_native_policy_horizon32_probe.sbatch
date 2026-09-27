#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-horizon32-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-horizon32-probe-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-horizon32-probe-$job
mkdir -p "$out"
printf "%s  %s\n" fbf8e4a12d96da6bf2fb32e63b10916b7d3699c360b5d33ad2b1a3493a9c3509 \
    "$disk/launch_puffer_calibrated_transfer.py" | sha256sum -c -
printf "%s  %s\n" 9f59b9047fa642c8dd4815f268d60cadba67fb691480ee82dd9d68f6de806626 \
    "$disk/run_puffer_calibrated_transfer.sh" | sha256sum -c -
printf "%s  %s\n" 5a2142491aad55cf4b2961f0c21572c80a80b76760d7ba6a4eccacf194b07371 \
    "$disk/monitor_coworld_steady_interval_v4.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
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
record["total_timesteps"] = 134_217_728
record["seed"] = 1338
record["initialize"] = {
    "run": "/recovery/device-context-native-policy-8b-25600/run",
    "checkpoint": "/recovery/device-context-native-policy-8b-25600/run/checkpoints/metta_generals/run/0000001073741824.bin",
    "sha256": "8553d1d6deab145bec2f4e7361d02d1de8c6df434ba9e85b13cf86bdc17921b2",
    "allow_published_checkpoint": True,
}
assert record["overrides"]["train.gamma"] == .999
record["overrides"].update({
    "vec.total_agents": 65536, "vec.num_buffers": 1, "vec.num_threads": 1,
    "base.cudagraphs": -1, "base.checkpoint_interval": 16,
    "policy.hidden_size": 128, "policy.num_layers": 4,
    "train.learning_rate": 0.003, "train.ent_coef": 0.0, "train.replay_ratio": 4.0, "train.horizon": 32, "train.minibatch_size": 65536,
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
name=relh-classic-native-policy-horizon32-probe-$job
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
) > "$disk/native-policy-horizon32-probe-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-policy-horizon32-probe-pilot-$job-0"
timeout -k 30s 25m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
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
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /recovery/run_puffer_calibrated_transfer.sh train \
    --config /recovery-output/config.json \
    --build /recovery/device-context-native-policy-65536-build-25526/build \
    --output /recovery-output/run > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-policy-horizon32-probe 600 2097152; then
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
assert completed["trained_timesteps"] == 134_217_728
checkpoint = run / completed["final_checkpoint"]
data = checkpoint.read_bytes()
assert len(data) > 0 and len(data) % 4 == 0
assert all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", data))
print("NATIVE_PARAMETERS_FINITE", len(data) // 4, hashlib.sha256(data).hexdigest(), flush=True)
PY
fi
exit "$status"
'

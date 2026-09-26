#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-hintreuse-8192-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/device-hintreuse-8192-pilot-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/device-hintreuse-8192-pilot-$job
mkdir -p "$out"
printf "%s  %s\n" 7020609f826afdb5574ad5ef983196598564e1da895927d65603eb4d051d612a \
    "$w/device-hintreuse-8192-build-17977/puffer" | sha256sum -c -
printf "%s  %s\n" 8dc2689bea60e46b0089ec613f69877daefd813f15236460159d7990febcbaef \
    "$w/metta-v11-transfer-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" 285f1782bed8450601fe59d5884710c193c3a4c8ceb807f9e00edfbc00fe00e8 \
    "$w/metta_puffer-hintreuse.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
python3 - "$w/device-hintreuse-8192-build-17977/build.json" <<"PY"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())["config"]
options = config["python_environment"]["options"]
assert options["opponent"] == "strong_mixed"
assert options["teacher"] == "expander_harvester"
assert options["supervise_teacher"] and options["imitation_weight"] == 0.0
assert options["expander_hint_features"] and options["prior_hint_features"]
phase = config["fabric"]["teacher"]["phases"][0]
assert (phase["ppo_coefficient"], phase["coefficient"], phase["action_mix"]) == (1.0, 0.25, 0.0)
PY
python3 - "$w/normalized-clean-pilot-16788/config.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
record["total_timesteps"] = 4_194_304
record["initialize"] = {
    "run": "/work/v11-transfer-source-run",
    "checkpoint": "/work/v11-transfer-source-run/checkpoints/metta_generals/run/0000000031457280.bin",
    "sha256": "939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499",
    "allow_environment_transfer": True,
    "allow_policy_only_transfer": True,
}
record["seed"] = 1304
record["overrides"].update({
    "vec.total_agents": 8192, "vec.num_buffers": 1, "vec.num_threads": 1,
    "base.cudagraphs": -1, "base.checkpoint_interval": 16,
    "train.learning_rate": 0.0001,
})
Path(sys.argv[2]).write_text(json.dumps(record) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
gpu_local=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; gpu_local=$i; break; fi
done
test -n "$gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$gpu local_index=$gpu_local"
name=relh-classic-device-hintreuse-8192-pilot-$job
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$w/device-hintreuse-8192-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$w/device-hintreuse-8192-pilot-$job-0"
timeout -k 30s 16m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-hintreuse.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-v11-transfer-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh train \
    --config "/work/device-hintreuse-8192-pilot-$job/config.json" \
    --build /work/device-hintreuse-8192-build-17977 \
    --output "/work/device-hintreuse-8192-pilot-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval_600.py" \
        "$w" "$job" "$((now-start))" device-hintreuse-8192 600; then
        echo "Device bridge smoke stopped by startup or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

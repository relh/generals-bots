#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-calibrated-aligned-discount-serving-probe
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-calibrated-aligned-discount-serving-probe-%j.log
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
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" eb034d4dcf79dd4c4a16d65595c3195cba676c06b008003af34c7cedef469502 \
    "$disk/metta_puffer-calibrated-v1.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-calibrated-aligned-discount-serving-probe-$job
mkdir -p "$out"
printf "%s  %s\n" 9b6f6e82e293fffa6786906bff0b45da9de7e6ae694351d5b23f4c276a87247f \
    "$disk/evaluate_coworld_frozen_greedy.py" | sha256sum -c -
printf "%s  %s\n" 9f0da8ff1d0d8a50a614d5fdb54f4fdf17ba5cac7a3b99a4b619efd7b4673d59 \
    "$disk/launch_puffer_calibrated_greedy.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
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
name=relh-classic-device-context-calibrated-aligned-discount-serving-probe-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
run=device-context-calibrated-aligned-discount-134m-25118
checkpoint=0000000134217728.bin
digest=$(python3 - "$disk/$run/run" <<"PY"
import hashlib, json, sys
from pathlib import Path
run = Path(sys.argv[1])
completed = json.loads((run / "completed.json").read_text())
assert completed["trained_timesteps"] == 134217728
assert Path(completed["final_checkpoint"]).name == "0000000134217728.bin"
print(hashlib.sha256((run / completed["final_checkpoint"]).read_bytes()).hexdigest())
PY
)
for item in \
    "a92b4f3232903984f76292ccb71590e0d5450255166f9793a826d4986d3d68e1 neural_codec-calibrated-serving-20260927.py" \
    "b2ab08723a4636aa431686ea0c6fd56bacccd02d235df9e091be880048b8968c neural_player-calibrated-serving-20260927.py" \
    "3762d8f33e5ea16c67c44c038fc5953c4f629a63cb26eafda0dbe5a538e291eb probe_context_frozen_bundle-serving-20260927.py"; do
    read -r checksum filename <<< "$item"
    printf "%s  %s\n" "$checksum" "$disk/$filename" | sha256sum -c -
done
common=(--rm --name "$name" --gpus "device=$gpu" --cpus=16
    --ulimit core=0:0 --user "$(id -u):$(id -g)"
    -v "$w:/work" -v "$out:/recovery-output" -v "$disk:/recovery"
    -v /var/tmp/relh-generals-jax-cache:/cache
    -v "$disk/metta_puffer-calibrated-v1.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro"
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro"
    -v "$disk/neural_codec-calibrated-serving-20260927.py:/work/source-normalized-16245/integrations/softmax/neural_codec.py:ro"
    -v "$disk/neural_player-calibrated-serving-20260927.py:/work/source-normalized-16245/integrations/softmax/neural_player.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache)
timeout -k 30s 2m docker run "${common[@]}" \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    relh-generals-b300:20260923b python -m pip install --no-cache-dir \
    --target /recovery-output/deps websockets==16.0 > "$out/deps.log" 2>&1
# CPU inference here deliberately measures the CPU hosted-serving path, not training SPS.
timeout -k 30s 8m docker run "${common[@]}" -e JAX_PLATFORMS=cpu \
    -e PYTHONPATH=/recovery-output/deps:/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    relh-generals-b300:20260923b python /recovery/probe_context_frozen_bundle-serving-20260927.py \
    --build /recovery/device-context-calibrated-aligned-discount-16384-build-25098/build/build.json \
    --checkpoint "/recovery/$run/run/checkpoints/metta_generals/run/$checkpoint" \
    --sha256 "$digest" --output /recovery-output/probe > "$out/probe.log" 2>&1
cat "$out/probe/probe.json"
'

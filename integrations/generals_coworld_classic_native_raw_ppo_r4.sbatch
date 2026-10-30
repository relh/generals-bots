#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-ppo-r4
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-ppo-r4-%j.log
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

python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_raw_ppo-r4-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCByZXdhcmQtb25seSBQdWZmZXI1IHRyYWluaW5nIG9uIHB1YmxpYyBDb3dvcmxkIENsYXNzaWMgb2JzZXJ2YXRpb25zLiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBtYXRoCmltcG9ydCBzdHJ1Y3QKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIG1ldHRhX3RyYWluaW5nLnB1ZmZlciBpbXBvcnQgUnVuQ29uZmlnLCB0cmFpbl9wdWZmZXIKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10ZW1wbGF0ZSIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdGltZXN0ZXBzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OF8zODhfNjA4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTQwMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbGVhcm5pbmctcmF0ZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9LjAwMykKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tcmVwbGF5LXJhdGlvIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0xLjApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoZWNrcG9pbnQtaW50ZXJ2YWwiLCB0eXBlPWludCwgZGVmYXVsdD0xKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmIChhcmdzLnRpbWVzdGVwcyA8PSAwIG9yIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MubGVhcm5pbmdfcmF0ZSkgb3IgYXJncy5sZWFybmluZ19yYXRlIDw9IDAKICAgICAgICAgICAgb3Igbm90IG1hdGguaXNmaW5pdGUoYXJncy5yZXBsYXlfcmF0aW8pIG9yIGFyZ3MucmVwbGF5X3JhdGlvIDw9IDAKICAgICAgICAgICAgb3IgYXJncy5jaGVja3BvaW50X2ludGVydmFsIDw9IDApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIkludmFsaWQgdHJhaW5pbmcgYnVkZ2V0IG9yIGxlYXJuaW5nIHJhdGUiKQogICAgYnVpbGQgPSBqc29uLmxvYWRzKChhcmdzLmJ1aWxkIC8gImJ1aWxkLmpzb24iKS5yZWFkX3RleHQoKSkKICAgIG9wdGlvbnMgPSBidWlsZFsiY29uZmlnIl1bInB5dGhvbl9lbnZpcm9ubWVudCJdWyJvcHRpb25zIl0KICAgIGlmIChidWlsZFsiY29uZmlnIl1bIm5hdGl2ZV9oaW50X3ByaW9yIl0gaXMgbm90IE5vbmUgb3Igb3B0aW9uc1siaGludF9mZWF0dXJlcyJdCiAgICAgICAgICAgIG9yIG9wdGlvbnNbImNvbXBhY3RfZmVhdHVyZXMiXSBvciBvcHRpb25zWyJwcmlvcl9oaW50X2ZlYXR1cmVzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1sic3VwZXJ2aXNlX3RlYWNoZXIiXSBvciBvcHRpb25zWyJ0ZWFjaGVyX3JvbGxvdXRzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1sic2hhcGluZ19nYW1tYSJdICE9IC45OTkgb3Igbm90IG9wdGlvbnNbImJhbGFuY2Vfb3Bwb25lbnRfc2lkZXMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJvcHBvbmVudCJdICE9ICJzdHJvbmdfbWl4ZWQiKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIHJhdyBSTCBlbnZpcm9ubWVudCIpCiAgICByZWNvcmQgPSBqc29uLmxvYWRzKGFyZ3MudGVtcGxhdGUucmVhZF90ZXh0KCkpCiAgICByZWNvcmRbInRvdGFsX3RpbWVzdGVwcyJdID0gYXJncy50aW1lc3RlcHMKICAgIHJlY29yZFsic2VlZCJdID0gYXJncy5zZWVkCiAgICByZWNvcmRbImluaXRpYWxpemUiXSA9IE5vbmUKICAgIHJlY29yZFsibmF0aXZlX3BvbGljeV9pbml0aWFsaXplciJdID0gTm9uZQogICAgcmVjb3JkWyJvdmVycmlkZXMiXS51cGRhdGUoewogICAgICAgICJ2ZWMudG90YWxfYWdlbnRzIjogNjU1MzYsICJ2ZWMubnVtX2J1ZmZlcnMiOiAxLCAidmVjLm51bV90aHJlYWRzIjogMSwKICAgICAgICAiYmFzZS5jdWRhZ3JhcGhzIjogLTEsICJiYXNlLmNoZWNrcG9pbnRfaW50ZXJ2YWwiOiBhcmdzLmNoZWNrcG9pbnRfaW50ZXJ2YWwsCiAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwgInBvbGljeS5udW1fbGF5ZXJzIjogMCwKICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IGFyZ3MubGVhcm5pbmdfcmF0ZSwgInRyYWluLmVudF9jb2VmIjogLjAxLAogICAgICAgICJ0cmFpbi5yZXBsYXlfcmF0aW8iOiBhcmdzLnJlcGxheV9yYXRpbywgInRyYWluLmhvcml6b24iOiAxNiwKICAgICAgICAidHJhaW4ubWluaWJhdGNoX3NpemUiOiA1MjQyODgsCiAgICB9KQogICAgYXNzZXJ0IHJlY29yZFsib3ZlcnJpZGVzIl1bInRyYWluLmdhbW1hIl0gPT0gLjk5OQogICAgY29uZmlnID0gUnVuQ29uZmlnLm1vZGVsX3ZhbGlkYXRlKHJlY29yZCkKICAgIChhcmdzLm91dHB1dCAvICJjb25maWcuanNvbiIpLndyaXRlX3RleHQoY29uZmlnLm1vZGVsX2R1bXBfanNvbihpbmRlbnQ9MikgKyAiXG4iKQogICAgcmVzdWx0ID0gdHJhaW5fcHVmZmVyKGFyZ3MuYnVpbGQsIGFyZ3Mub3V0cHV0IC8gInJ1biIsIGNvbmZpZykKICAgIGFzc2VydCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMgPT0gYXJncy50aW1lc3RlcHMKICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSAoNTEyICogNjE3NCArIDE3NjggKiA1MTIpICogNAogICAgYXNzZXJ0IGFsbChtYXRoLmlzZmluaXRlKHZhbHVlWzBdKSBmb3IgdmFsdWUgaW4gc3RydWN0Lml0ZXJfdW5wYWNrKCI8ZiIsIGRhdGEpKQogICAgcHJpbnQoIk5BVElWRV9SQVdfUFBPX1BJTE9UX09LIiwgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzLAogICAgICAgICAgaGFzaGxpYi5zaGEyNTYoZGF0YSkuaGV4ZGlnZXN0KCksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/native-raw-ppo-r4-$job
mkdir -p "$out"
printf '%s  %s\n' 481d6618cc36689a663cf63b6ec7b754089f79e6eac2f760e3afc69d8b6d24ec "$disk/train_native_raw_ppo-r4-20260927.py" | sha256sum -c -
python3 - "$disk/native-raw-ppo-build-26769/build/build.json" <<'PYINNER'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); manifest=json.loads(path.read_text()); config=manifest['config']
assert hashlib.sha256(path.with_name('puffer').read_bytes()).hexdigest()==manifest['binary_sha256']
assert config['native_hint_prior'] is None and config['fabric'] is None
assert config['python_environment']['options']['shaping_gamma']==.999
assert config['python_environment']['options']['balance_opponent_sides']
PYINNER
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-native-raw-ppo-r4-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-raw-ppo-r4-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-raw-ppo-r4-pilot-$job-0"
timeout -k 30s 20m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output" \
    -v "$disk/native-hint-prior-puffer.py-v2-20260927:/work/metta-device-wide-source/metta_training/puffer.py:ro" \
    -v "$disk/native-hint-prior-model_config.py-v2-20260927:/work/metta-device-wide-source/metta_training/model_config.py:ro" \
    -v "$disk/native-hint-prior-native_build.py-v2-20260927:/work/metta-device-wide-source/metta_training/native_build.py:ro" \
    -v "$disk/native-hint-prior-hint_prior.cuh-v2-20260927:/work/metta-device-wide-source/metta_training/native/hint_prior.cuh:ro" \
    -v "$disk/native_hint_initializer-feedforward-20260927.py:/work/source-normalized-16245/integrations/native_hint_initializer.py:ro" \
    -v "$disk/metta_puffer-terminal-boundary-fixed-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b python /recovery/train_native_raw_ppo-r4-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-raw-ppo-build-26769/build \
      --output /recovery-output --timesteps 8388608 --replay-ratio 4 --checkpoint-interval 1 > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-raw-ppo-r4 600 1048576 30000; then
        echo "Raw PPO replay-ratio pilot stopped by startup or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
NODE

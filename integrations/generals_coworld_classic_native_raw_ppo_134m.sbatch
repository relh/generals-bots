#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-ppo-134m
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-ppo-134m-%j.log
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
path=Path(sys.argv[1])/'train_native_raw_ppo-134m-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCByZXdhcmQtb25seSBQdWZmZXI1IHRyYWluaW5nIG9uIHB1YmxpYyBDb3dvcmxkIENsYXNzaWMgb2JzZXJ2YXRpb25zLiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBtYXRoCmltcG9ydCBzdHJ1Y3QKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIG1ldHRhX3RyYWluaW5nLnB1ZmZlciBpbXBvcnQgUnVuQ29uZmlnLCB0cmFpbl9wdWZmZXIKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10ZW1wbGF0ZSIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdGltZXN0ZXBzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OF8zODhfNjA4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTQwMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbGVhcm5pbmctcmF0ZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9LjAwMykKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tY2hlY2twb2ludC1pbnRlcnZhbCIsIHR5cGU9aW50LCBkZWZhdWx0PTEpCiAgICBhcmdzID0gcGFyc2VyLnBhcnNlX2FyZ3MoKQogICAgaWYgKGFyZ3MudGltZXN0ZXBzIDw9IDAgb3Igbm90IG1hdGguaXNmaW5pdGUoYXJncy5sZWFybmluZ19yYXRlKSBvciBhcmdzLmxlYXJuaW5nX3JhdGUgPD0gMAogICAgICAgICAgICBvciBhcmdzLmNoZWNrcG9pbnRfaW50ZXJ2YWwgPD0gMCk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiSW52YWxpZCB0cmFpbmluZyBidWRnZXQgb3IgbGVhcm5pbmcgcmF0ZSIpCiAgICBidWlsZCA9IGpzb24ubG9hZHMoKGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIpLnJlYWRfdGV4dCgpKQogICAgb3B0aW9ucyA9IGJ1aWxkWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQogICAgaWYgKGJ1aWxkWyJjb25maWciXVsibmF0aXZlX2hpbnRfcHJpb3IiXSBpcyBub3QgTm9uZSBvciBvcHRpb25zWyJoaW50X2ZlYXR1cmVzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1siY29tcGFjdF9mZWF0dXJlcyJdIG9yIG9wdGlvbnNbInByaW9yX2hpbnRfZmVhdHVyZXMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJzdXBlcnZpc2VfdGVhY2hlciJdIG9yIG9wdGlvbnNbInRlYWNoZXJfcm9sbG91dHMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJzaGFwaW5nX2dhbW1hIl0gIT0gLjk5OSBvciBub3Qgb3B0aW9uc1siYmFsYW5jZV9vcHBvbmVudF9zaWRlcyJdCiAgICAgICAgICAgIG9yIG9wdGlvbnNbIm9wcG9uZW50Il0gIT0gInN0cm9uZ19taXhlZCIpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlVuZXhwZWN0ZWQgcmF3IFJMIGVudmlyb25tZW50IikKICAgIHJlY29yZCA9IGpzb24ubG9hZHMoYXJncy50ZW1wbGF0ZS5yZWFkX3RleHQoKSkKICAgIHJlY29yZFsidG90YWxfdGltZXN0ZXBzIl0gPSBhcmdzLnRpbWVzdGVwcwogICAgcmVjb3JkWyJzZWVkIl0gPSBhcmdzLnNlZWQKICAgIHJlY29yZFsiaW5pdGlhbGl6ZSJdID0gTm9uZQogICAgcmVjb3JkWyJuYXRpdmVfcG9saWN5X2luaXRpYWxpemVyIl0gPSBOb25lCiAgICByZWNvcmRbIm92ZXJyaWRlcyJdLnVwZGF0ZSh7CiAgICAgICAgInZlYy50b3RhbF9hZ2VudHMiOiA2NTUzNiwgInZlYy5udW1fYnVmZmVycyI6IDEsICJ2ZWMubnVtX3RocmVhZHMiOiAxLAogICAgICAgICJiYXNlLmN1ZGFncmFwaHMiOiAtMSwgImJhc2UuY2hlY2twb2ludF9pbnRlcnZhbCI6IGFyZ3MuY2hlY2twb2ludF9pbnRlcnZhbCwKICAgICAgICAicG9saWN5LmhpZGRlbl9zaXplIjogNTEyLCAicG9saWN5Lm51bV9sYXllcnMiOiAwLAogICAgICAgICJ0cmFpbi5sZWFybmluZ19yYXRlIjogYXJncy5sZWFybmluZ19yYXRlLCAidHJhaW4uZW50X2NvZWYiOiAuMDEsCiAgICAgICAgInRyYWluLnJlcGxheV9yYXRpbyI6IDEuMCwgInRyYWluLmhvcml6b24iOiAxNiwKICAgICAgICAidHJhaW4ubWluaWJhdGNoX3NpemUiOiA1MjQyODgsCiAgICB9KQogICAgYXNzZXJ0IHJlY29yZFsib3ZlcnJpZGVzIl1bInRyYWluLmdhbW1hIl0gPT0gLjk5OQogICAgY29uZmlnID0gUnVuQ29uZmlnLm1vZGVsX3ZhbGlkYXRlKHJlY29yZCkKICAgIChhcmdzLm91dHB1dCAvICJjb25maWcuanNvbiIpLndyaXRlX3RleHQoY29uZmlnLm1vZGVsX2R1bXBfanNvbihpbmRlbnQ9MikgKyAiXG4iKQogICAgcmVzdWx0ID0gdHJhaW5fcHVmZmVyKGFyZ3MuYnVpbGQsIGFyZ3Mub3V0cHV0IC8gInJ1biIsIGNvbmZpZykKICAgIGFzc2VydCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMgPT0gYXJncy50aW1lc3RlcHMKICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSAoNTEyICogNjE3NCArIDE3NjggKiA1MTIpICogNAogICAgYXNzZXJ0IGFsbChtYXRoLmlzZmluaXRlKHZhbHVlWzBdKSBmb3IgdmFsdWUgaW4gc3RydWN0Lml0ZXJfdW5wYWNrKCI8ZiIsIGRhdGEpKQogICAgcHJpbnQoIk5BVElWRV9SQVdfUFBPX1BJTE9UX09LIiwgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzLAogICAgICAgICAgaGFzaGxpYi5zaGEyNTYoZGF0YSkuaGV4ZGlnZXN0KCksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
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
out=$disk/native-raw-ppo-134m-$job
mkdir -p "$out"
printf '%s  %s\n' e2f98e49c3441bfae1b3dc40734cc26a26a7f222f63054b99c7f7058783e9950 "$disk/train_native_raw_ppo-134m-20260927.py" | sha256sum -c -
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
name=relh-classic-native-raw-ppo-134m-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-raw-ppo-134m-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-raw-ppo-134m-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_raw_ppo-134m-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-raw-ppo-build-26769/build \
      --output /recovery-output --timesteps 134217728 --checkpoint-interval 16 > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-fixed-prior-pilot 600 1048576 30000; then
        echo "Fixed-prior pilot stopped by startup or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
NODE

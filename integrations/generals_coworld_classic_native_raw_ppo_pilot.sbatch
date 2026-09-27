#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-ppo-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-ppo-pilot-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_raw_ppo-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCByZXdhcmQtb25seSBQdWZmZXI1IHRyYWluaW5nIG9uIHB1YmxpYyBDb3dvcmxkIENsYXNzaWMgb2JzZXJ2YXRpb25zLiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBtYXRoCmltcG9ydCBzdHJ1Y3QKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIG1ldHRhX3RyYWluaW5nLnB1ZmZlciBpbXBvcnQgUnVuQ29uZmlnLCB0cmFpbl9wdWZmZXIKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10ZW1wbGF0ZSIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdGltZXN0ZXBzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OF8zODhfNjA4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTQwMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbGVhcm5pbmctcmF0ZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9LjAwMykKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBpZiBhcmdzLnRpbWVzdGVwcyA8PSAwIG9yIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MubGVhcm5pbmdfcmF0ZSkgb3IgYXJncy5sZWFybmluZ19yYXRlIDw9IDA6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiSW52YWxpZCB0cmFpbmluZyBidWRnZXQgb3IgbGVhcm5pbmcgcmF0ZSIpCiAgICBidWlsZCA9IGpzb24ubG9hZHMoKGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIpLnJlYWRfdGV4dCgpKQogICAgb3B0aW9ucyA9IGJ1aWxkWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQogICAgaWYgKGJ1aWxkWyJjb25maWciXVsibmF0aXZlX2hpbnRfcHJpb3IiXSBpcyBub3QgTm9uZSBvciBvcHRpb25zWyJoaW50X2ZlYXR1cmVzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1siY29tcGFjdF9mZWF0dXJlcyJdIG9yIG9wdGlvbnNbInByaW9yX2hpbnRfZmVhdHVyZXMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJzdXBlcnZpc2VfdGVhY2hlciJdIG9yIG9wdGlvbnNbInRlYWNoZXJfcm9sbG91dHMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJzaGFwaW5nX2dhbW1hIl0gIT0gLjk5OSBvciBub3Qgb3B0aW9uc1siYmFsYW5jZV9vcHBvbmVudF9zaWRlcyJdCiAgICAgICAgICAgIG9yIG9wdGlvbnNbIm9wcG9uZW50Il0gIT0gInN0cm9uZ19taXhlZCIpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlVuZXhwZWN0ZWQgcmF3IFJMIGVudmlyb25tZW50IikKICAgIHJlY29yZCA9IGpzb24ubG9hZHMoYXJncy50ZW1wbGF0ZS5yZWFkX3RleHQoKSkKICAgIHJlY29yZFsidG90YWxfdGltZXN0ZXBzIl0gPSBhcmdzLnRpbWVzdGVwcwogICAgcmVjb3JkWyJzZWVkIl0gPSBhcmdzLnNlZWQKICAgIHJlY29yZFsiaW5pdGlhbGl6ZSJdID0gTm9uZQogICAgcmVjb3JkWyJuYXRpdmVfcG9saWN5X2luaXRpYWxpemVyIl0gPSBOb25lCiAgICByZWNvcmRbIm92ZXJyaWRlcyJdLnVwZGF0ZSh7CiAgICAgICAgInZlYy50b3RhbF9hZ2VudHMiOiA2NTUzNiwgInZlYy5udW1fYnVmZmVycyI6IDEsICJ2ZWMubnVtX3RocmVhZHMiOiAxLAogICAgICAgICJiYXNlLmN1ZGFncmFwaHMiOiAtMSwgImJhc2UuY2hlY2twb2ludF9pbnRlcnZhbCI6IDEsCiAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwgInBvbGljeS5udW1fbGF5ZXJzIjogMCwKICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IGFyZ3MubGVhcm5pbmdfcmF0ZSwgInRyYWluLmVudF9jb2VmIjogLjAxLAogICAgICAgICJ0cmFpbi5yZXBsYXlfcmF0aW8iOiAxLjAsICJ0cmFpbi5ob3Jpem9uIjogMTYsCiAgICAgICAgInRyYWluLm1pbmliYXRjaF9zaXplIjogNTI0Mjg4LAogICAgfSkKICAgIGFzc2VydCByZWNvcmRbIm92ZXJyaWRlcyJdWyJ0cmFpbi5nYW1tYSJdID09IC45OTkKICAgIGNvbmZpZyA9IFJ1bkNvbmZpZy5tb2RlbF92YWxpZGF0ZShyZWNvcmQpCiAgICAoYXJncy5vdXRwdXQgLyAiY29uZmlnLmpzb24iKS53cml0ZV90ZXh0KGNvbmZpZy5tb2RlbF9kdW1wX2pzb24oaW5kZW50PTIpICsgIlxuIikKICAgIHJlc3VsdCA9IHRyYWluX3B1ZmZlcihhcmdzLmJ1aWxkLCBhcmdzLm91dHB1dCAvICJydW4iLCBjb25maWcpCiAgICBhc3NlcnQgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzID09IGFyZ3MudGltZXN0ZXBzCiAgICBjaGVja3BvaW50ID0gYXJncy5vdXRwdXQgLyAicnVuIiAvIHJlc3VsdC5maW5hbF9jaGVja3BvaW50CiAgICBkYXRhID0gY2hlY2twb2ludC5yZWFkX2J5dGVzKCkKICAgIGFzc2VydCBsZW4oZGF0YSkgPT0gKDUxMiAqIDYxNzQgKyAxNzY4ICogNTEyKSAqIDQKICAgIGFzc2VydCBhbGwobWF0aC5pc2Zpbml0ZSh2YWx1ZVswXSkgZm9yIHZhbHVlIGluIHN0cnVjdC5pdGVyX3VucGFjaygiPGYiLCBkYXRhKSkKICAgIHByaW50KCJOQVRJVkVfUkFXX1BQT19QSUxPVF9PSyIsIHJlc3VsdC50cmFpbmVkX3RpbWVzdGVwcywKICAgICAgICAgIGhhc2hsaWIuc2hhMjU2KGRhdGEpLmhleGRpZ2VzdCgpLCBmbHVzaD1UcnVlKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/native-raw-ppo-pilot-$job
mkdir -p "$out"
printf '%s  %s\n' 7fb6e2568b531b7ae6ef55bf968f1fc94815dbefcd342c0dde5b130fcfc61e69 "$disk/train_native_raw_ppo-20260927.py" | sha256sum -c -
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
name=relh-classic-native-raw-ppo-pilot-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-raw-ppo-pilot-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-raw-ppo-pilot-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_raw_ppo-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-raw-ppo-build-26769/build \
      --output /recovery-output > "$out/container.log" 2>&1 &
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

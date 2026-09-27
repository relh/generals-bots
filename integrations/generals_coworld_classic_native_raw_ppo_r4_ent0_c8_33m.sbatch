#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-ppo-r4-ent0-c8-33m
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-ppo-r4-ent0-c8-33m-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_raw_ppo-r4-ent0-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCByZXdhcmQtb25seSBQdWZmZXI1IHRyYWluaW5nIG9uIHB1YmxpYyBDb3dvcmxkIENsYXNzaWMgb2JzZXJ2YXRpb25zLiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBtYXRoCmltcG9ydCBzdHJ1Y3QKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIG1ldHRhX3RyYWluaW5nLnB1ZmZlciBpbXBvcnQgUnVuQ29uZmlnLCB0cmFpbl9wdWZmZXIKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10ZW1wbGF0ZSIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdGltZXN0ZXBzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OF8zODhfNjA4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTQwMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbGVhcm5pbmctcmF0ZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9LjAwMykKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tZW50cm9weS1jb2VmIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0uMDEpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXJlcGxheS1yYXRpbyIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MS4wKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1jaGVja3BvaW50LWludGVydmFsIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MSkKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBpZiAoYXJncy50aW1lc3RlcHMgPD0gMCBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLmxlYXJuaW5nX3JhdGUpIG9yIGFyZ3MubGVhcm5pbmdfcmF0ZSA8PSAwCiAgICAgICAgICAgIG9yIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MuZW50cm9weV9jb2VmKSBvciBhcmdzLmVudHJvcHlfY29lZiA8IDAKICAgICAgICAgICAgb3Igbm90IG1hdGguaXNmaW5pdGUoYXJncy5yZXBsYXlfcmF0aW8pIG9yIGFyZ3MucmVwbGF5X3JhdGlvIDw9IDAKICAgICAgICAgICAgb3IgYXJncy5jaGVja3BvaW50X2ludGVydmFsIDw9IDApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIkludmFsaWQgdHJhaW5pbmcgYnVkZ2V0IG9yIGxlYXJuaW5nIHJhdGUiKQogICAgYnVpbGQgPSBqc29uLmxvYWRzKChhcmdzLmJ1aWxkIC8gImJ1aWxkLmpzb24iKS5yZWFkX3RleHQoKSkKICAgIG9wdGlvbnMgPSBidWlsZFsiY29uZmlnIl1bInB5dGhvbl9lbnZpcm9ubWVudCJdWyJvcHRpb25zIl0KICAgIGlmIChidWlsZFsiY29uZmlnIl1bIm5hdGl2ZV9oaW50X3ByaW9yIl0gaXMgbm90IE5vbmUgb3Igb3B0aW9uc1siaGludF9mZWF0dXJlcyJdCiAgICAgICAgICAgIG9yIG9wdGlvbnNbImNvbXBhY3RfZmVhdHVyZXMiXSBvciBvcHRpb25zWyJwcmlvcl9oaW50X2ZlYXR1cmVzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1sic3VwZXJ2aXNlX3RlYWNoZXIiXSBvciBvcHRpb25zWyJ0ZWFjaGVyX3JvbGxvdXRzIl0KICAgICAgICAgICAgb3Igb3B0aW9uc1sic2hhcGluZ19nYW1tYSJdICE9IC45OTkgb3Igbm90IG9wdGlvbnNbImJhbGFuY2Vfb3Bwb25lbnRfc2lkZXMiXQogICAgICAgICAgICBvciBvcHRpb25zWyJvcHBvbmVudCJdICE9ICJzdHJvbmdfbWl4ZWQiKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIHJhdyBSTCBlbnZpcm9ubWVudCIpCiAgICByZWNvcmQgPSBqc29uLmxvYWRzKGFyZ3MudGVtcGxhdGUucmVhZF90ZXh0KCkpCiAgICByZWNvcmRbInRvdGFsX3RpbWVzdGVwcyJdID0gYXJncy50aW1lc3RlcHMKICAgIHJlY29yZFsic2VlZCJdID0gYXJncy5zZWVkCiAgICByZWNvcmRbImluaXRpYWxpemUiXSA9IE5vbmUKICAgIHJlY29yZFsibmF0aXZlX3BvbGljeV9pbml0aWFsaXplciJdID0gTm9uZQogICAgcmVjb3JkWyJvdmVycmlkZXMiXS51cGRhdGUoewogICAgICAgICJ2ZWMudG90YWxfYWdlbnRzIjogNjU1MzYsICJ2ZWMubnVtX2J1ZmZlcnMiOiAxLCAidmVjLm51bV90aHJlYWRzIjogMSwKICAgICAgICAiYmFzZS5jdWRhZ3JhcGhzIjogLTEsICJiYXNlLmNoZWNrcG9pbnRfaW50ZXJ2YWwiOiBhcmdzLmNoZWNrcG9pbnRfaW50ZXJ2YWwsCiAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwgInBvbGljeS5udW1fbGF5ZXJzIjogMCwKICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IGFyZ3MubGVhcm5pbmdfcmF0ZSwgInRyYWluLmFubmVhbF9sciI6IDAsCiAgICAgICAgInRyYWluLmVudF9jb2VmIjogYXJncy5lbnRyb3B5X2NvZWYsCiAgICAgICAgInRyYWluLnJlcGxheV9yYXRpbyI6IGFyZ3MucmVwbGF5X3JhdGlvLCAidHJhaW4uaG9yaXpvbiI6IDE2LAogICAgICAgICJ0cmFpbi5taW5pYmF0Y2hfc2l6ZSI6IDUyNDI4OCwKICAgIH0pCiAgICBhc3NlcnQgcmVjb3JkWyJvdmVycmlkZXMiXVsidHJhaW4uZ2FtbWEiXSA9PSAuOTk5CiAgICBjb25maWcgPSBSdW5Db25maWcubW9kZWxfdmFsaWRhdGUocmVjb3JkKQogICAgKGFyZ3Mub3V0cHV0IC8gImNvbmZpZy5qc29uIikud3JpdGVfdGV4dChjb25maWcubW9kZWxfZHVtcF9qc29uKGluZGVudD0yKSArICJcbiIpCiAgICByZXN1bHQgPSB0cmFpbl9wdWZmZXIoYXJncy5idWlsZCwgYXJncy5vdXRwdXQgLyAicnVuIiwgY29uZmlnKQogICAgYXNzZXJ0IHJlc3VsdC50cmFpbmVkX3RpbWVzdGVwcyA9PSBhcmdzLnRpbWVzdGVwcwogICAgY2hlY2twb2ludCA9IGFyZ3Mub3V0cHV0IC8gInJ1biIgLyByZXN1bHQuZmluYWxfY2hlY2twb2ludAogICAgZGF0YSA9IGNoZWNrcG9pbnQucmVhZF9ieXRlcygpCiAgICBhc3NlcnQgbGVuKGRhdGEpID09ICg1MTIgKiA2MTc0ICsgMTc2OCAqIDUxMikgKiA0CiAgICBhc3NlcnQgYWxsKG1hdGguaXNmaW5pdGUodmFsdWVbMF0pIGZvciB2YWx1ZSBpbiBzdHJ1Y3QuaXRlcl91bnBhY2soIjxmIiwgZGF0YSkpCiAgICBwcmludCgiTkFUSVZFX1JBV19QUE9fUElMT1RfT0siLCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMsCiAgICAgICAgICBoYXNobGliLnNoYTI1NihkYXRhKS5oZXhkaWdlc3QoKSwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg==')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/native-raw-ppo-r4-ent0-c8-33m-$job
mkdir -p "$out"
printf '%s  %s\n' 0221f099d7dbc6a1bcf5be865aac9329ae3fc31c023c5320a25773a220925e66 "$disk/train_native_raw_ppo-r4-ent0-20260927.py" | sha256sum -c -
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
name=relh-classic-native-raw-ppo-r4-ent0-c8-33m-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-raw-ppo-r4-ent0-c8-33m-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-raw-ppo-r4-ent0-c8-33m-pilot-$job-0"
timeout -k 30s 20m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
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
    relh-generals-b300:20260923b python /recovery/train_native_raw_ppo-r4-ent0-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-raw-ppo-build-26769/build \
      --output /recovery-output --timesteps 33554432 --replay-ratio 4 --entropy-coef 0 --checkpoint-interval 8 > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-raw-ppo-r4-ent0-c8-33m 600 1048576 30000; then
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

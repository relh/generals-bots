#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-random-r4-ent0-33m
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-random-r4-ent0-33m-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_raw_ppo-random-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCByZXdhcmQtb25seSBQdWZmZXI1IHRyYWluaW5nIG9uIHB1YmxpYyBDb3dvcmxkIENsYXNzaWMgb2JzZXJ2YXRpb25zLiIiIgoKaW1wb3J0IGFyZ3BhcnNlCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmltcG9ydCBtYXRoCmltcG9ydCBzdHJ1Y3QKZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgpmcm9tIG1ldHRhX3RyYWluaW5nLnB1ZmZlciBpbXBvcnQgUnVuQ29uZmlnLCB0cmFpbl9wdWZmZXIKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10ZW1wbGF0ZSIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdGltZXN0ZXBzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OF8zODhfNjA4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTQwMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbGVhcm5pbmctcmF0ZSIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MC4wMDMpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWVudHJvcHktY29lZiIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MC4wMSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tcmVwbGF5LXJhdGlvIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0xLjApCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoZWNrcG9pbnQtaW50ZXJ2YWwiLCB0eXBlPWludCwgZGVmYXVsdD0xKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1vcHBvbmVudCIsIGNob2ljZXM9KCJzdHJvbmdfbWl4ZWQiLCAicmFuZG9tIiksIGRlZmF1bHQ9InN0cm9uZ19taXhlZCIpCiAgICBhcmdzID0gcGFyc2VyLnBhcnNlX2FyZ3MoKQogICAgaWYgKAogICAgICAgIGFyZ3MudGltZXN0ZXBzIDw9IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLmxlYXJuaW5nX3JhdGUpCiAgICAgICAgb3IgYXJncy5sZWFybmluZ19yYXRlIDw9IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLmVudHJvcHlfY29lZikKICAgICAgICBvciBhcmdzLmVudHJvcHlfY29lZiA8IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLnJlcGxheV9yYXRpbykKICAgICAgICBvciBhcmdzLnJlcGxheV9yYXRpbyA8PSAwCiAgICAgICAgb3IgYXJncy5jaGVja3BvaW50X2ludGVydmFsIDw9IDAKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiSW52YWxpZCB0cmFpbmluZyBidWRnZXQgb3IgbGVhcm5pbmcgcmF0ZSIpCiAgICBidWlsZCA9IGpzb24ubG9hZHMoKGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIpLnJlYWRfdGV4dCgpKQogICAgb3B0aW9ucyA9IGJ1aWxkWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQogICAgaWYgKAogICAgICAgIGJ1aWxkWyJjb25maWciXVsibmF0aXZlX2hpbnRfcHJpb3IiXSBpcyBub3QgTm9uZQogICAgICAgIG9yIG9wdGlvbnNbImhpbnRfZmVhdHVyZXMiXQogICAgICAgIG9yIG9wdGlvbnNbImNvbXBhY3RfZmVhdHVyZXMiXQogICAgICAgIG9yIG9wdGlvbnNbInByaW9yX2hpbnRfZmVhdHVyZXMiXQogICAgICAgIG9yIG9wdGlvbnNbInN1cGVydmlzZV90ZWFjaGVyIl0KICAgICAgICBvciBvcHRpb25zWyJ0ZWFjaGVyX3JvbGxvdXRzIl0KICAgICAgICBvciBvcHRpb25zWyJzaGFwaW5nX2dhbW1hIl0gIT0gMC45OTkKICAgICAgICBvciBub3Qgb3B0aW9uc1siYmFsYW5jZV9vcHBvbmVudF9zaWRlcyJdCiAgICAgICAgb3Igb3B0aW9uc1sib3Bwb25lbnQiXSAhPSBhcmdzLm9wcG9uZW50CiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlVuZXhwZWN0ZWQgcmF3IFJMIGVudmlyb25tZW50IikKICAgIHJlY29yZCA9IGpzb24ubG9hZHMoYXJncy50ZW1wbGF0ZS5yZWFkX3RleHQoKSkKICAgIHJlY29yZFsidG90YWxfdGltZXN0ZXBzIl0gPSBhcmdzLnRpbWVzdGVwcwogICAgcmVjb3JkWyJzZWVkIl0gPSBhcmdzLnNlZWQKICAgIHJlY29yZFsiaW5pdGlhbGl6ZSJdID0gTm9uZQogICAgcmVjb3JkWyJuYXRpdmVfcG9saWN5X2luaXRpYWxpemVyIl0gPSBOb25lCiAgICByZWNvcmRbIm92ZXJyaWRlcyJdLnVwZGF0ZSgKICAgICAgICB7CiAgICAgICAgICAgICJ2ZWMudG90YWxfYWdlbnRzIjogNjU1MzYsCiAgICAgICAgICAgICJ2ZWMubnVtX2J1ZmZlcnMiOiAxLAogICAgICAgICAgICAidmVjLm51bV90aHJlYWRzIjogMSwKICAgICAgICAgICAgImJhc2UuY3VkYWdyYXBocyI6IC0xLAogICAgICAgICAgICAiYmFzZS5jaGVja3BvaW50X2ludGVydmFsIjogYXJncy5jaGVja3BvaW50X2ludGVydmFsLAogICAgICAgICAgICAicG9saWN5LmhpZGRlbl9zaXplIjogNTEyLAogICAgICAgICAgICAicG9saWN5Lm51bV9sYXllcnMiOiAwLAogICAgICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IGFyZ3MubGVhcm5pbmdfcmF0ZSwKICAgICAgICAgICAgInRyYWluLmFubmVhbF9sciI6IDAsCiAgICAgICAgICAgICJ0cmFpbi5lbnRfY29lZiI6IGFyZ3MuZW50cm9weV9jb2VmLAogICAgICAgICAgICAidHJhaW4ucmVwbGF5X3JhdGlvIjogYXJncy5yZXBsYXlfcmF0aW8sCiAgICAgICAgICAgICJ0cmFpbi5ob3Jpem9uIjogMTYsCiAgICAgICAgICAgICJ0cmFpbi5taW5pYmF0Y2hfc2l6ZSI6IDUyNDI4OCwKICAgICAgICB9CiAgICApCiAgICBhc3NlcnQgcmVjb3JkWyJvdmVycmlkZXMiXVsidHJhaW4uZ2FtbWEiXSA9PSAwLjk5OQogICAgY29uZmlnID0gUnVuQ29uZmlnLm1vZGVsX3ZhbGlkYXRlKHJlY29yZCkKICAgIChhcmdzLm91dHB1dCAvICJjb25maWcuanNvbiIpLndyaXRlX3RleHQoY29uZmlnLm1vZGVsX2R1bXBfanNvbihpbmRlbnQ9MikgKyAiXG4iKQogICAgcmVzdWx0ID0gdHJhaW5fcHVmZmVyKGFyZ3MuYnVpbGQsIGFyZ3Mub3V0cHV0IC8gInJ1biIsIGNvbmZpZykKICAgIGFzc2VydCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMgPT0gYXJncy50aW1lc3RlcHMKICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSAoNTEyICogNjE3NCArIDE3NjggKiA1MTIpICogNAogICAgYXNzZXJ0IGFsbChtYXRoLmlzZmluaXRlKHZhbHVlWzBdKSBmb3IgdmFsdWUgaW4gc3RydWN0Lml0ZXJfdW5wYWNrKCI8ZiIsIGRhdGEpKQogICAgcHJpbnQoIk5BVElWRV9SQVdfUFBPX1BJTE9UX09LIiwgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzLCBoYXNobGliLnNoYTI1NihkYXRhKS5oZXhkaWdlc3QoKSwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg==')
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
out=$disk/native-raw-random-r4-ent0-33m-$job
mkdir -p "$out"
printf '%s  %s\n' 92efd16ea7e3d26255d0ed95934218515800ddc5dd0c0cb88531fda2786dc8af "$disk/train_native_raw_ppo-random-20260927.py" | sha256sum -c -
python3 - "$disk/native-raw-random-build-26856/build/build.json" <<'PYINNER'
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
name=relh-classic-native-raw-random-r4-ent0-33m-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-raw-random-r4-ent0-33m-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-raw-random-r4-ent0-33m-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_raw_ppo-random-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-raw-random-build-26856/build \
      --output /recovery-output --timesteps 33554432 --replay-ratio 4 --entropy-coef 0 --checkpoint-interval 8 --opponent random > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-raw-random-r4-ent0-33m 600 1048576 30000; then
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

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-fixed-prior75-advnorm-r4-33m
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-fixed-prior75-advnorm-r4-33m-%j.log
set -euo pipefail
export TMPDIR=/var/tmp/relh-generals-recovery
mkdir -p "$TMPDIR"
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_fixed_prior-advnorm-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCBQdWZmZXI1IHRyYWluaW5nIGZyb20gYSBmaXhlZCBwdWJsaWMtaGludCBwcmlvciBhbmQgd2VhayB0cmFpbmFibGUgc2VlZC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgbWF0aAppbXBvcnQgc3RydWN0CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKZnJvbSBtZXR0YV90cmFpbmluZy5wdWZmZXIgaW1wb3J0IE5hdGl2ZVBvbGljeUluaXRpYWxpemF0aW9uLCBSdW5Db25maWcsIHRyYWluX3B1ZmZlcgoKZnJvbSBpbnRlZ3JhdGlvbnMubmF0aXZlX2hpbnRfaW5pdGlhbGl6ZXIgaW1wb3J0IGV4cG9ydAoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXRlbXBsYXRlIiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tb3V0cHV0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10aW1lc3RlcHMiLCB0eXBlPWludCwgZGVmYXVsdD04XzM4OF82MDgpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXNlZWQiLCB0eXBlPWludCwgZGVmYXVsdD0xMzk0KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sZWFybmluZy1yYXRlIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjAwMDEpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXJlcGxheS1yYXRpbyIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MS4wKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1jaGVja3BvaW50LWludGVydmFsIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm9ybWFsaXplLWFkdmFudGFnZXMiLCBhY3Rpb249InN0b3JlX3RydWUiKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmICgKICAgICAgICBhcmdzLnRpbWVzdGVwcyA8PSAwCiAgICAgICAgb3Igbm90IG1hdGguaXNmaW5pdGUoYXJncy5sZWFybmluZ19yYXRlKQogICAgICAgIG9yIGFyZ3MubGVhcm5pbmdfcmF0ZSA8IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLnJlcGxheV9yYXRpbykKICAgICAgICBvciBhcmdzLnJlcGxheV9yYXRpbyA8PSAwCiAgICAgICAgb3IgYXJncy5jaGVja3BvaW50X2ludGVydmFsIDw9IDAKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiSW52YWxpZCB0cmFpbmluZyBidWRnZXQgb3IgbGVhcm5pbmcgcmF0ZSIpCiAgICBidWlsZCA9IGpzb24ubG9hZHMoKGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIpLnJlYWRfdGV4dCgpKQogICAgaWYgYXJncy5ub3JtYWxpemVfYWR2YW50YWdlcyBhbmQgIm5vcm1fYWR2ID0gMCIgbm90IGluIChhcmdzLmJ1aWxkIC8gInNvdXJjZS9jb25maWcvZGVmYXVsdC5pbmkiKS5yZWFkX3RleHQoKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJOYXRpdmUgYWR2YW50YWdlIG5vcm1hbGl6YXRpb24gcmVxdWlyZXMgaXRzIHZlcmlmaWVkIGJ1aWxkIikKICAgIG9wdGlvbnMgPSBidWlsZFsiY29uZmlnIl1bInB5dGhvbl9lbnZpcm9ubWVudCJdWyJvcHRpb25zIl0KICAgIGlmICgKICAgICAgICBidWlsZFsiY29uZmlnIl1bIm5hdGl2ZV9oaW50X3ByaW9yIl0gaXMgTm9uZQogICAgICAgIG9yIG9wdGlvbnNbInNoYXBpbmdfZ2FtbWEiXSAhPSAwLjk5OQogICAgICAgIG9yIG5vdCBvcHRpb25zWyJiYWxhbmNlX29wcG9uZW50X3NpZGVzIl0KICAgICAgICBvciBvcHRpb25zWyJvcHBvbmVudCJdICE9ICJzdHJvbmdfbWl4ZWQiCiAgICApOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIlVuZXhwZWN0ZWQgZml4ZWQtcHJpb3IgdHJhaW5pbmcgZW52aXJvbm1lbnQiKQogICAgYXNzZXJ0IG5vdCAoYXJncy5vdXRwdXQgLyAicnVuIikuZXhpc3RzKCkKICAgIF8sIGluaXRpYWxpemVyID0gZXhwb3J0KGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIsIGFyZ3Mub3V0cHV0IC8gImluaXRpYWxpemVyIiwgc2NhbGU9MC4wMDEsIGxheWVycz0wKQogICAgYXJ0aWZhY3QgPSBhcmdzLm91dHB1dCAvICJpbml0aWFsaXplci9pbml0aWFsaXplci5qc29uIgogICAgcmVjb3JkID0ganNvbi5sb2FkcyhhcmdzLnRlbXBsYXRlLnJlYWRfdGV4dCgpKQogICAgcmVjb3JkWyJ0b3RhbF90aW1lc3RlcHMiXSA9IGFyZ3MudGltZXN0ZXBzCiAgICByZWNvcmRbInNlZWQiXSA9IGFyZ3Muc2VlZAogICAgcmVjb3JkWyJpbml0aWFsaXplIl0gPSBOb25lCiAgICByZWNvcmRbIm5hdGl2ZV9wb2xpY3lfaW5pdGlhbGl6ZXIiXSA9IGRpY3QoCiAgICAgICAgbWFuaWZlc3Q9c3RyKGFydGlmYWN0KSwgc2hhMjU2PWhhc2hsaWIuc2hhMjU2KGFydGlmYWN0LnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgICkKICAgIHJlY29yZFsib3ZlcnJpZGVzIl0udXBkYXRlKAogICAgICAgIHsKICAgICAgICAgICAgInZlYy50b3RhbF9hZ2VudHMiOiA2NTUzNiwKICAgICAgICAgICAgInZlYy5udW1fYnVmZmVycyI6IDEsCiAgICAgICAgICAgICJ2ZWMubnVtX3RocmVhZHMiOiAxLAogICAgICAgICAgICAiYmFzZS5jdWRhZ3JhcGhzIjogLTEsCiAgICAgICAgICAgICJiYXNlLmNoZWNrcG9pbnRfaW50ZXJ2YWwiOiBhcmdzLmNoZWNrcG9pbnRfaW50ZXJ2YWwsCiAgICAgICAgICAgICJwb2xpY3kuaGlkZGVuX3NpemUiOiA1MTIsCiAgICAgICAgICAgICJwb2xpY3kubnVtX2xheWVycyI6IDAsCiAgICAgICAgICAgICJ0cmFpbi5sZWFybmluZ19yYXRlIjogYXJncy5sZWFybmluZ19yYXRlLAogICAgICAgICAgICAidHJhaW4uYW5uZWFsX2xyIjogMCwKICAgICAgICAgICAgInRyYWluLmVudF9jb2VmIjogMC4wLAogICAgICAgICAgICAidHJhaW4ucmVwbGF5X3JhdGlvIjogYXJncy5yZXBsYXlfcmF0aW8sCiAgICAgICAgICAgICJ0cmFpbi5ob3Jpem9uIjogMTYsCiAgICAgICAgICAgICJ0cmFpbi5taW5pYmF0Y2hfc2l6ZSI6IDUyNDI4OCwKICAgICAgICB9CiAgICApCiAgICBpZiBhcmdzLm5vcm1hbGl6ZV9hZHZhbnRhZ2VzOgogICAgICAgIHJlY29yZFsib3ZlcnJpZGVzIl1bInRyYWluLm5vcm1fYWR2Il0gPSAxCiAgICBhc3NlcnQgcmVjb3JkWyJvdmVycmlkZXMiXVsidHJhaW4uZ2FtbWEiXSA9PSAwLjk5OQogICAgY29uZmlnID0gUnVuQ29uZmlnLm1vZGVsX3ZhbGlkYXRlKHJlY29yZCkKICAgIGFzc2VydCBpc2luc3RhbmNlKGNvbmZpZy5uYXRpdmVfcG9saWN5X2luaXRpYWxpemVyLCBOYXRpdmVQb2xpY3lJbml0aWFsaXphdGlvbikKICAgIChhcmdzLm91dHB1dCAvICJjb25maWcuanNvbiIpLndyaXRlX3RleHQoY29uZmlnLm1vZGVsX2R1bXBfanNvbihpbmRlbnQ9MikgKyAiXG4iKQogICAgcmVzdWx0ID0gdHJhaW5fcHVmZmVyKGFyZ3MuYnVpbGQsIGFyZ3Mub3V0cHV0IC8gInJ1biIsIGNvbmZpZykKICAgIGFzc2VydCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMgPT0gYXJncy50aW1lc3RlcHMKICAgIGNvcGllZF9zaGEgPSBoYXNobGliLnNoYTI1NigoYXJncy5vdXRwdXQgLyAicnVuL2luaXRpYWwtcG9saWN5LmJpbiIpLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgIGFzc2VydCBjb3BpZWRfc2hhID09IGluaXRpYWxpemVyWyJwb2xpY3lfc2hhMjU2Il0KICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSBpbml0aWFsaXplclsicGFyYW1ldGVyX2NvdW50Il0gKiA0CiAgICBhc3NlcnQgYWxsKG1hdGguaXNmaW5pdGUodmFsdWVbMF0pIGZvciB2YWx1ZSBpbiBzdHJ1Y3QuaXRlcl91bnBhY2soIjxmIiwgZGF0YSkpCiAgICBwcmludCgiTkFUSVZFX0ZJWEVEX1BSSU9SX1BJTE9UX09LIiwgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzLCBoYXNobGliLnNoYTI1NihkYXRhKS5oZXhkaWdlc3QoKSwgZmx1c2g9VHJ1ZSkKCgppZiBfX25hbWVfXyA9PSAiX19tYWluX18iOgogICAgbWFpbigpCg==')
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
out=$disk/native-fixed-prior75-advnorm-r4-33m-$job
mkdir -p "$out"
printf '%s  %s\n' 63e153faa7676f14028d4a157837c3bc7f2e768f8cb6075a3367f6ed5274f0a4 "$disk/train_native_fixed_prior-advnorm-20260927.py" | sha256sum -c -
python3 - "$disk/native-fixed-prior75-advnorm-build-v1-27009/build/build.json" <<'PYINNER'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); manifest=json.loads(path.read_text()); config=manifest['config']
assert hashlib.sha256(path.with_name('puffer').read_bytes()).hexdigest()==manifest['binary_sha256']
assert config['native_hint_prior']['cells']==441 and config['fabric'] is None
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
name=relh-classic-native-fixed-prior75-advnorm-r4-33m-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-fixed-prior75-advnorm-r4-33m-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-fixed-prior75-advnorm-r4-33m-pilot-$job-0"
timeout -k 30s 20m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output" \
    -v "$disk/native-hint-prior-puffer.py-v2-20260927:/work/metta-device-wide-source/metta_training/puffer.py:ro" \
    -v "$disk/native-hint-prior-model_config.py-v2-20260927:/work/metta-device-wide-source/metta_training/model_config.py:ro" \
    -v "$disk/native-advnorm-native_build-v2-20260927.py:/work/metta-device-wide-source/metta_training/native_build.py:ro" \
    -v "$disk/native-advnorm-learner-python-v1-20260927.py:/work/metta-device-wide-source/metta_training/learner.py:ro" \
    -v "$disk/native-advnorm-advantage.cuh-20260927:/work/metta-device-wide-source/metta_training/native/advantage.cuh:ro" \
    -v "$disk/native-advnorm-learner-v3-20260927.cuh:/work/metta-device-wide-source/metta_training/native/learner.cuh:ro" \
    -v "$disk/native-advnorm-replay-v2-20260927.cuh:/work/metta-device-wide-source/metta_training/native/replay.cuh:ro" \
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
    relh-generals-b300:20260923b python /recovery/train_native_fixed_prior-advnorm-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-fixed-prior75-advnorm-build-v1-27009/build \
      --output /recovery-output --learning-rate .003 --timesteps 33554432 --replay-ratio 4 --checkpoint-interval 8 --normalize-advantages > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-fixed-prior75-advnorm-r4-33m 600 1048576 30000; then
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

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-fixed-prior-lr3e3-134m
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-fixed-prior-lr3e3-134m-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_fixed_prior-lr3e3-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCBQdWZmZXI1IHRyYWluaW5nIGZyb20gYSBmaXhlZCBwdWJsaWMtaGludCBwcmlvciBhbmQgd2VhayB0cmFpbmFibGUgc2VlZC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgbWF0aAppbXBvcnQgc3RydWN0CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKZnJvbSBtZXR0YV90cmFpbmluZy5wdWZmZXIgaW1wb3J0IE5hdGl2ZVBvbGljeUluaXRpYWxpemF0aW9uLCBSdW5Db25maWcsIHRyYWluX3B1ZmZlcgoKZnJvbSBpbnRlZ3JhdGlvbnMubmF0aXZlX2hpbnRfaW5pdGlhbGl6ZXIgaW1wb3J0IGV4cG9ydAoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXRlbXBsYXRlIiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tb3V0cHV0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10aW1lc3RlcHMiLCB0eXBlPWludCwgZGVmYXVsdD04XzM4OF82MDgpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXNlZWQiLCB0eXBlPWludCwgZGVmYXVsdD0xMzk0KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sZWFybmluZy1yYXRlIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0uMDAwMSkKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBpZiBhcmdzLnRpbWVzdGVwcyA8PSAwIG9yIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MubGVhcm5pbmdfcmF0ZSkgb3IgYXJncy5sZWFybmluZ19yYXRlIDw9IDA6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiSW52YWxpZCB0cmFpbmluZyBidWRnZXQgb3IgbGVhcm5pbmcgcmF0ZSIpCiAgICBidWlsZCA9IGpzb24ubG9hZHMoKGFyZ3MuYnVpbGQgLyAiYnVpbGQuanNvbiIpLnJlYWRfdGV4dCgpKQogICAgb3B0aW9ucyA9IGJ1aWxkWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQogICAgaWYgKGJ1aWxkWyJjb25maWciXVsibmF0aXZlX2hpbnRfcHJpb3IiXSBpcyBOb25lIG9yIG9wdGlvbnNbInNoYXBpbmdfZ2FtbWEiXSAhPSAuOTk5CiAgICAgICAgICAgIG9yIG5vdCBvcHRpb25zWyJiYWxhbmNlX29wcG9uZW50X3NpZGVzIl0gb3Igb3B0aW9uc1sib3Bwb25lbnQiXSAhPSAic3Ryb25nX21peGVkIik6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiVW5leHBlY3RlZCBmaXhlZC1wcmlvciB0cmFpbmluZyBlbnZpcm9ubWVudCIpCiAgICBhc3NlcnQgbm90IChhcmdzLm91dHB1dCAvICJydW4iKS5leGlzdHMoKQogICAgXywgaW5pdGlhbGl6ZXIgPSBleHBvcnQoYXJncy5idWlsZCAvICJidWlsZC5qc29uIiwgYXJncy5vdXRwdXQgLyAiaW5pdGlhbGl6ZXIiLCBzY2FsZT0uMDAxLCBsYXllcnM9MCkKICAgIGFydGlmYWN0ID0gYXJncy5vdXRwdXQgLyAiaW5pdGlhbGl6ZXIvaW5pdGlhbGl6ZXIuanNvbiIKICAgIHJlY29yZCA9IGpzb24ubG9hZHMoYXJncy50ZW1wbGF0ZS5yZWFkX3RleHQoKSkKICAgIHJlY29yZFsidG90YWxfdGltZXN0ZXBzIl0gPSBhcmdzLnRpbWVzdGVwcwogICAgcmVjb3JkWyJzZWVkIl0gPSBhcmdzLnNlZWQKICAgIHJlY29yZFsiaW5pdGlhbGl6ZSJdID0gTm9uZQogICAgcmVjb3JkWyJuYXRpdmVfcG9saWN5X2luaXRpYWxpemVyIl0gPSBkaWN0KAogICAgICAgIG1hbmlmZXN0PXN0cihhcnRpZmFjdCksIHNoYTI1Nj1oYXNobGliLnNoYTI1NihhcnRpZmFjdC5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICApCiAgICByZWNvcmRbIm92ZXJyaWRlcyJdLnVwZGF0ZSh7CiAgICAgICAgInZlYy50b3RhbF9hZ2VudHMiOiA2NTUzNiwgInZlYy5udW1fYnVmZmVycyI6IDEsICJ2ZWMubnVtX3RocmVhZHMiOiAxLAogICAgICAgICJiYXNlLmN1ZGFncmFwaHMiOiAtMSwgImJhc2UuY2hlY2twb2ludF9pbnRlcnZhbCI6IDEsCiAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwgInBvbGljeS5udW1fbGF5ZXJzIjogMCwKICAgICAgICAidHJhaW4ubGVhcm5pbmdfcmF0ZSI6IGFyZ3MubGVhcm5pbmdfcmF0ZSwgInRyYWluLmVudF9jb2VmIjogMC4wLAogICAgICAgICJ0cmFpbi5yZXBsYXlfcmF0aW8iOiAxLjAsICJ0cmFpbi5ob3Jpem9uIjogMTYsCiAgICAgICAgInRyYWluLm1pbmliYXRjaF9zaXplIjogNTI0Mjg4LAogICAgfSkKICAgIGFzc2VydCByZWNvcmRbIm92ZXJyaWRlcyJdWyJ0cmFpbi5nYW1tYSJdID09IC45OTkKICAgIGNvbmZpZyA9IFJ1bkNvbmZpZy5tb2RlbF92YWxpZGF0ZShyZWNvcmQpCiAgICBhc3NlcnQgaXNpbnN0YW5jZShjb25maWcubmF0aXZlX3BvbGljeV9pbml0aWFsaXplciwgTmF0aXZlUG9saWN5SW5pdGlhbGl6YXRpb24pCiAgICAoYXJncy5vdXRwdXQgLyAiY29uZmlnLmpzb24iKS53cml0ZV90ZXh0KGNvbmZpZy5tb2RlbF9kdW1wX2pzb24oaW5kZW50PTIpICsgIlxuIikKICAgIHJlc3VsdCA9IHRyYWluX3B1ZmZlcihhcmdzLmJ1aWxkLCBhcmdzLm91dHB1dCAvICJydW4iLCBjb25maWcpCiAgICBhc3NlcnQgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzID09IGFyZ3MudGltZXN0ZXBzCiAgICBjb3BpZWRfc2hhID0gaGFzaGxpYi5zaGEyNTYoKGFyZ3Mub3V0cHV0IC8gInJ1bi9pbml0aWFsLXBvbGljeS5iaW4iKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICBhc3NlcnQgY29waWVkX3NoYSA9PSBpbml0aWFsaXplclsicG9saWN5X3NoYTI1NiJdCiAgICBjaGVja3BvaW50ID0gYXJncy5vdXRwdXQgLyAicnVuIiAvIHJlc3VsdC5maW5hbF9jaGVja3BvaW50CiAgICBkYXRhID0gY2hlY2twb2ludC5yZWFkX2J5dGVzKCkKICAgIGFzc2VydCBsZW4oZGF0YSkgPT0gaW5pdGlhbGl6ZXJbInBhcmFtZXRlcl9jb3VudCJdICogNAogICAgYXNzZXJ0IGFsbChtYXRoLmlzZmluaXRlKHZhbHVlWzBdKSBmb3IgdmFsdWUgaW4gc3RydWN0Lml0ZXJfdW5wYWNrKCI8ZiIsIGRhdGEpKQogICAgcHJpbnQoIk5BVElWRV9GSVhFRF9QUklPUl9QSUxPVF9PSyIsIHJlc3VsdC50cmFpbmVkX3RpbWVzdGVwcywKICAgICAgICAgIGhhc2hsaWIuc2hhMjU2KGRhdGEpLmhleGRpZ2VzdCgpLCBmbHVzaD1UcnVlKQoKCmlmIF9fbmFtZV9fID09ICJfX21haW5fXyI6CiAgICBtYWluKCkK')
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
out=$disk/native-fixed-prior-lr3e3-134m-$job
mkdir -p "$out"
printf '%s  %s\n' 2fb3a127d73d924eee33ce3511c843aab8a8964aa30a732b9fa54842c6cce11e "$disk/train_native_fixed_prior-lr3e3-20260927.py" | sha256sum -c -
python3 - "$disk/native-hint-prior-build-v2-26757/build/build.json" <<'PYINNER'
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
name=relh-classic-native-fixed-prior-lr3e3-134m-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-fixed-prior-lr3e3-134m-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-fixed-prior-lr3e3-134m-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_fixed_prior-lr3e3-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-hint-prior-build-v2-26757/build \
      --output /recovery-output --learning-rate .003 --timesteps 134217728 > "$out/container.log" 2>&1 &
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

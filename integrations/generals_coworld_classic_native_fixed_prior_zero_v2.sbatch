#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-fixed-prior-zero-v2
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-fixed-prior-zero-v2-%j.log
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
path=Path(sys.argv[1])/'train_native_fixed_prior-zero-v2-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCBQdWZmZXI1IHRyYWluaW5nIGZyb20gYSBmaXhlZCBwdWJsaWMtaGludCBwcmlvciBhbmQgd2VhayB0cmFpbmFibGUgc2VlZC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgbWF0aAppbXBvcnQgc3RydWN0CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKZnJvbSBtZXR0YV90cmFpbmluZy5wdWZmZXIgaW1wb3J0IE5hdGl2ZVBvbGljeUluaXRpYWxpemF0aW9uLCBSdW5Db25maWcsIHRyYWluX3B1ZmZlcgoKZnJvbSBpbnRlZ3JhdGlvbnMubmF0aXZlX2hpbnRfaW5pdGlhbGl6ZXIgaW1wb3J0IGV4cG9ydAoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXRlbXBsYXRlIiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tb3V0cHV0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10aW1lc3RlcHMiLCB0eXBlPWludCwgZGVmYXVsdD04XzM4OF82MDgpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXNlZWQiLCB0eXBlPWludCwgZGVmYXVsdD0xMzk0KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sZWFybmluZy1yYXRlIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0uMDAwMSkKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBpZiBhcmdzLnRpbWVzdGVwcyA8PSAwIG9yIG5vdCBtYXRoLmlzZmluaXRlKGFyZ3MubGVhcm5pbmdfcmF0ZSkgb3IgYXJncy5sZWFybmluZ19yYXRlIDwgMDoKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJJbnZhbGlkIHRyYWluaW5nIGJ1ZGdldCBvciBsZWFybmluZyByYXRlIikKICAgIGJ1aWxkID0ganNvbi5sb2FkcygoYXJncy5idWlsZCAvICJidWlsZC5qc29uIikucmVhZF90ZXh0KCkpCiAgICBvcHRpb25zID0gYnVpbGRbImNvbmZpZyJdWyJweXRob25fZW52aXJvbm1lbnQiXVsib3B0aW9ucyJdCiAgICBpZiAoYnVpbGRbImNvbmZpZyJdWyJuYXRpdmVfaGludF9wcmlvciJdIGlzIE5vbmUgb3Igb3B0aW9uc1sic2hhcGluZ19nYW1tYSJdICE9IC45OTkKICAgICAgICAgICAgb3Igbm90IG9wdGlvbnNbImJhbGFuY2Vfb3Bwb25lbnRfc2lkZXMiXSBvciBvcHRpb25zWyJvcHBvbmVudCJdICE9ICJzdHJvbmdfbWl4ZWQiKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIGZpeGVkLXByaW9yIHRyYWluaW5nIGVudmlyb25tZW50IikKICAgIGFzc2VydCBub3QgKGFyZ3Mub3V0cHV0IC8gInJ1biIpLmV4aXN0cygpCiAgICBfLCBpbml0aWFsaXplciA9IGV4cG9ydChhcmdzLmJ1aWxkIC8gImJ1aWxkLmpzb24iLCBhcmdzLm91dHB1dCAvICJpbml0aWFsaXplciIsIHNjYWxlPS4wMDEsIGxheWVycz0wKQogICAgYXJ0aWZhY3QgPSBhcmdzLm91dHB1dCAvICJpbml0aWFsaXplci9pbml0aWFsaXplci5qc29uIgogICAgcmVjb3JkID0ganNvbi5sb2FkcyhhcmdzLnRlbXBsYXRlLnJlYWRfdGV4dCgpKQogICAgcmVjb3JkWyJ0b3RhbF90aW1lc3RlcHMiXSA9IGFyZ3MudGltZXN0ZXBzCiAgICByZWNvcmRbInNlZWQiXSA9IGFyZ3Muc2VlZAogICAgcmVjb3JkWyJpbml0aWFsaXplIl0gPSBOb25lCiAgICByZWNvcmRbIm5hdGl2ZV9wb2xpY3lfaW5pdGlhbGl6ZXIiXSA9IGRpY3QoCiAgICAgICAgbWFuaWZlc3Q9c3RyKGFydGlmYWN0KSwgc2hhMjU2PWhhc2hsaWIuc2hhMjU2KGFydGlmYWN0LnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgICkKICAgIHJlY29yZFsib3ZlcnJpZGVzIl0udXBkYXRlKHsKICAgICAgICAidmVjLnRvdGFsX2FnZW50cyI6IDY1NTM2LCAidmVjLm51bV9idWZmZXJzIjogMSwgInZlYy5udW1fdGhyZWFkcyI6IDEsCiAgICAgICAgImJhc2UuY3VkYWdyYXBocyI6IC0xLCAiYmFzZS5jaGVja3BvaW50X2ludGVydmFsIjogMSwKICAgICAgICAicG9saWN5LmhpZGRlbl9zaXplIjogNTEyLCAicG9saWN5Lm51bV9sYXllcnMiOiAwLAogICAgICAgICJ0cmFpbi5sZWFybmluZ19yYXRlIjogYXJncy5sZWFybmluZ19yYXRlLCAidHJhaW4uZW50X2NvZWYiOiAwLjAsCiAgICAgICAgInRyYWluLnJlcGxheV9yYXRpbyI6IDEuMCwgInRyYWluLmhvcml6b24iOiAxNiwKICAgICAgICAidHJhaW4ubWluaWJhdGNoX3NpemUiOiA1MjQyODgsCiAgICB9KQogICAgYXNzZXJ0IHJlY29yZFsib3ZlcnJpZGVzIl1bInRyYWluLmdhbW1hIl0gPT0gLjk5OQogICAgY29uZmlnID0gUnVuQ29uZmlnLm1vZGVsX3ZhbGlkYXRlKHJlY29yZCkKICAgIGFzc2VydCBpc2luc3RhbmNlKGNvbmZpZy5uYXRpdmVfcG9saWN5X2luaXRpYWxpemVyLCBOYXRpdmVQb2xpY3lJbml0aWFsaXphdGlvbikKICAgIChhcmdzLm91dHB1dCAvICJjb25maWcuanNvbiIpLndyaXRlX3RleHQoY29uZmlnLm1vZGVsX2R1bXBfanNvbihpbmRlbnQ9MikgKyAiXG4iKQogICAgcmVzdWx0ID0gdHJhaW5fcHVmZmVyKGFyZ3MuYnVpbGQsIGFyZ3Mub3V0cHV0IC8gInJ1biIsIGNvbmZpZykKICAgIGFzc2VydCByZXN1bHQudHJhaW5lZF90aW1lc3RlcHMgPT0gYXJncy50aW1lc3RlcHMKICAgIGNvcGllZF9zaGEgPSBoYXNobGliLnNoYTI1NigoYXJncy5vdXRwdXQgLyAicnVuL2luaXRpYWwtcG9saWN5LmJpbiIpLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkKICAgIGFzc2VydCBjb3BpZWRfc2hhID09IGluaXRpYWxpemVyWyJwb2xpY3lfc2hhMjU2Il0KICAgIGNoZWNrcG9pbnQgPSBhcmdzLm91dHB1dCAvICJydW4iIC8gcmVzdWx0LmZpbmFsX2NoZWNrcG9pbnQKICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgYXNzZXJ0IGxlbihkYXRhKSA9PSBpbml0aWFsaXplclsicGFyYW1ldGVyX2NvdW50Il0gKiA0CiAgICBhc3NlcnQgYWxsKG1hdGguaXNmaW5pdGUodmFsdWVbMF0pIGZvciB2YWx1ZSBpbiBzdHJ1Y3QuaXRlcl91bnBhY2soIjxmIiwgZGF0YSkpCiAgICBwcmludCgiTkFUSVZFX0ZJWEVEX1BSSU9SX1BJTE9UX09LIiwgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzLAogICAgICAgICAgaGFzaGxpYi5zaGEyNTYoZGF0YSkuaGV4ZGlnZXN0KCksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
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
out=$disk/native-fixed-prior-zero-v2-$job
mkdir -p "$out"
printf '%s  %s\n' 3509209c37b93f8fd7faac2dc6d76bdf7b40e29865e8db5990f40c74fcb77677 "$disk/train_native_fixed_prior-zero-v2-20260927.py" | sha256sum -c -
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
name=relh-classic-native-fixed-prior-zero-v2-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-fixed-prior-zero-v2-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-fixed-prior-zero-v2-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_fixed_prior-zero-v2-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-hint-prior-build-v2-26757/build \
      --output /recovery-output --timesteps 1048576 --learning-rate 0 > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-fixed-prior-zero 600 1048576 30000; then
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

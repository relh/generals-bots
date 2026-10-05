#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-fixed-prior75-gae99-h64-33m-v2
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-fixed-prior75-gae99-h64-33m-v2-%j.log
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

export TMPDIR=/var/tmp/relh-generals-recovery
mkdir -p "$TMPDIR"
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'train_native_fixed_prior-gae99-20260927.py'
data=base64.b64decode('IiIiQm91bmRlZCBQdWZmZXI1IHRyYWluaW5nIGZyb20gYSBmaXhlZCBwdWJsaWMtaGludCBwcmlvciBhbmQgd2VhayB0cmFpbmFibGUgc2VlZC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgppbXBvcnQgbWF0aAppbXBvcnQgc3RydWN0CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKZnJvbSBtZXR0YV90cmFpbmluZy5wdWZmZXIgaW1wb3J0IE5hdGl2ZVBvbGljeUluaXRpYWxpemF0aW9uLCBSdW5Db25maWcsIHRyYWluX3B1ZmZlcgoKZnJvbSBpbnRlZ3JhdGlvbnMubmF0aXZlX2hpbnRfaW5pdGlhbGl6ZXIgaW1wb3J0IGV4cG9ydAoKCmRlZiBtYWluKCk6CiAgICBwYXJzZXIgPSBhcmdwYXJzZS5Bcmd1bWVudFBhcnNlcigpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXRlbXBsYXRlIiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tb3V0cHV0IiwgdHlwZT1QYXRoLCByZXF1aXJlZD1UcnVlKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10aW1lc3RlcHMiLCB0eXBlPWludCwgZGVmYXVsdD04XzM4OF82MDgpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXNlZWQiLCB0eXBlPWludCwgZGVmYXVsdD0xMzk0KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1sZWFybmluZy1yYXRlIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjAwMDEpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXJlcGxheS1yYXRpbyIsIHR5cGU9ZmxvYXQsIGRlZmF1bHQ9MS4wKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1jaGVja3BvaW50LWludGVydmFsIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tbm9ybWFsaXplLWFkdmFudGFnZXMiLCBhY3Rpb249InN0b3JlX3RydWUiKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS10b3RhbC1hZ2VudHMiLCB0eXBlPWludCwgZGVmYXVsdD02NTUzNikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0taG9yaXpvbiIsIHR5cGU9aW50LCBkZWZhdWx0PTE2KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1nYWUtbGFtYmRhIiwgdHlwZT1mbG9hdCwgZGVmYXVsdD0wLjkwKQogICAgYXJncyA9IHBhcnNlci5wYXJzZV9hcmdzKCkKICAgIGlmICgKICAgICAgICBhcmdzLnRpbWVzdGVwcyA8PSAwCiAgICAgICAgb3Igbm90IG1hdGguaXNmaW5pdGUoYXJncy5sZWFybmluZ19yYXRlKQogICAgICAgIG9yIGFyZ3MubGVhcm5pbmdfcmF0ZSA8IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLnJlcGxheV9yYXRpbykKICAgICAgICBvciBhcmdzLnJlcGxheV9yYXRpbyA8PSAwCiAgICAgICAgb3IgYXJncy5jaGVja3BvaW50X2ludGVydmFsIDw9IDAKICAgICAgICBvciBhcmdzLnRvdGFsX2FnZW50cyA8PSAwCiAgICAgICAgb3IgYXJncy5ob3Jpem9uIDw9IDAKICAgICAgICBvciBub3QgbWF0aC5pc2Zpbml0ZShhcmdzLmdhZV9sYW1iZGEpCiAgICAgICAgb3Igbm90IDAgPCBhcmdzLmdhZV9sYW1iZGEgPD0gMQogICAgKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJJbnZhbGlkIHRyYWluaW5nIGJ1ZGdldCBvciBsZWFybmluZyByYXRlIikKICAgIGJ1aWxkID0ganNvbi5sb2FkcygoYXJncy5idWlsZCAvICJidWlsZC5qc29uIikucmVhZF90ZXh0KCkpCiAgICBpZiBhcmdzLm5vcm1hbGl6ZV9hZHZhbnRhZ2VzIGFuZCAibm9ybV9hZHYgPSAwIiBub3QgaW4gKGFyZ3MuYnVpbGQgLyAic291cmNlL2NvbmZpZy9kZWZhdWx0LmluaSIpLnJlYWRfdGV4dCgpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIk5hdGl2ZSBhZHZhbnRhZ2Ugbm9ybWFsaXphdGlvbiByZXF1aXJlcyBpdHMgdmVyaWZpZWQgYnVpbGQiKQogICAgb3B0aW9ucyA9IGJ1aWxkWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQogICAgaWYgKAogICAgICAgIGJ1aWxkWyJjb25maWciXVsibmF0aXZlX2hpbnRfcHJpb3IiXSBpcyBOb25lCiAgICAgICAgb3Igb3B0aW9uc1sic2hhcGluZ19nYW1tYSJdICE9IDAuOTk5CiAgICAgICAgb3Igbm90IG9wdGlvbnNbImJhbGFuY2Vfb3Bwb25lbnRfc2lkZXMiXQogICAgICAgIG9yIG9wdGlvbnNbIm9wcG9uZW50Il0gIT0gInN0cm9uZ19taXhlZCIKICAgICk6CiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiVW5leHBlY3RlZCBmaXhlZC1wcmlvciB0cmFpbmluZyBlbnZpcm9ubWVudCIpCiAgICBhc3NlcnQgbm90IChhcmdzLm91dHB1dCAvICJydW4iKS5leGlzdHMoKQogICAgXywgaW5pdGlhbGl6ZXIgPSBleHBvcnQoYXJncy5idWlsZCAvICJidWlsZC5qc29uIiwgYXJncy5vdXRwdXQgLyAiaW5pdGlhbGl6ZXIiLCBzY2FsZT0wLjAwMSwgbGF5ZXJzPTApCiAgICBhcnRpZmFjdCA9IGFyZ3Mub3V0cHV0IC8gImluaXRpYWxpemVyL2luaXRpYWxpemVyLmpzb24iCiAgICByZWNvcmQgPSBqc29uLmxvYWRzKGFyZ3MudGVtcGxhdGUucmVhZF90ZXh0KCkpCiAgICByZWNvcmRbInRvdGFsX3RpbWVzdGVwcyJdID0gYXJncy50aW1lc3RlcHMKICAgIHJlY29yZFsic2VlZCJdID0gYXJncy5zZWVkCiAgICByZWNvcmRbImluaXRpYWxpemUiXSA9IE5vbmUKICAgIHJlY29yZFsibmF0aXZlX3BvbGljeV9pbml0aWFsaXplciJdID0gZGljdCgKICAgICAgICBtYW5pZmVzdD1zdHIoYXJ0aWZhY3QpLCBzaGEyNTY9aGFzaGxpYi5zaGEyNTYoYXJ0aWZhY3QucmVhZF9ieXRlcygpKS5oZXhkaWdlc3QoKQogICAgKQogICAgcmVjb3JkWyJvdmVycmlkZXMiXS51cGRhdGUoCiAgICAgICAgewogICAgICAgICAgICAidmVjLnRvdGFsX2FnZW50cyI6IGFyZ3MudG90YWxfYWdlbnRzLAogICAgICAgICAgICAidmVjLm51bV9idWZmZXJzIjogMSwKICAgICAgICAgICAgInZlYy5udW1fdGhyZWFkcyI6IDEsCiAgICAgICAgICAgICJiYXNlLmN1ZGFncmFwaHMiOiAtMSwKICAgICAgICAgICAgImJhc2UuY2hlY2twb2ludF9pbnRlcnZhbCI6IGFyZ3MuY2hlY2twb2ludF9pbnRlcnZhbCwKICAgICAgICAgICAgInBvbGljeS5oaWRkZW5fc2l6ZSI6IDUxMiwKICAgICAgICAgICAgInBvbGljeS5udW1fbGF5ZXJzIjogMCwKICAgICAgICAgICAgInRyYWluLmxlYXJuaW5nX3JhdGUiOiBhcmdzLmxlYXJuaW5nX3JhdGUsCiAgICAgICAgICAgICJ0cmFpbi5hbm5lYWxfbHIiOiAwLAogICAgICAgICAgICAidHJhaW4uZW50X2NvZWYiOiAwLjAsCiAgICAgICAgICAgICJ0cmFpbi5yZXBsYXlfcmF0aW8iOiBhcmdzLnJlcGxheV9yYXRpbywKICAgICAgICAgICAgInRyYWluLmhvcml6b24iOiBhcmdzLmhvcml6b24sCiAgICAgICAgICAgICJ0cmFpbi5nYWVfbGFtYmRhIjogYXJncy5nYWVfbGFtYmRhLAogICAgICAgICAgICAidHJhaW4ubWluaWJhdGNoX3NpemUiOiA1MjQyODgsCiAgICAgICAgfQogICAgKQogICAgaWYgYXJncy5ub3JtYWxpemVfYWR2YW50YWdlczoKICAgICAgICByZWNvcmRbIm92ZXJyaWRlcyJdWyJ0cmFpbi5ub3JtX2FkdiJdID0gMQogICAgYXNzZXJ0IHJlY29yZFsib3ZlcnJpZGVzIl1bInRyYWluLmdhbW1hIl0gPT0gMC45OTkKICAgIGNvbmZpZyA9IFJ1bkNvbmZpZy5tb2RlbF92YWxpZGF0ZShyZWNvcmQpCiAgICBhc3NlcnQgaXNpbnN0YW5jZShjb25maWcubmF0aXZlX3BvbGljeV9pbml0aWFsaXplciwgTmF0aXZlUG9saWN5SW5pdGlhbGl6YXRpb24pCiAgICAoYXJncy5vdXRwdXQgLyAiY29uZmlnLmpzb24iKS53cml0ZV90ZXh0KGNvbmZpZy5tb2RlbF9kdW1wX2pzb24oaW5kZW50PTIpICsgIlxuIikKICAgIHJlc3VsdCA9IHRyYWluX3B1ZmZlcihhcmdzLmJ1aWxkLCBhcmdzLm91dHB1dCAvICJydW4iLCBjb25maWcpCiAgICBhc3NlcnQgcmVzdWx0LnRyYWluZWRfdGltZXN0ZXBzID09IGFyZ3MudGltZXN0ZXBzCiAgICBjb3BpZWRfc2hhID0gaGFzaGxpYi5zaGEyNTYoKGFyZ3Mub3V0cHV0IC8gInJ1bi9pbml0aWFsLXBvbGljeS5iaW4iKS5yZWFkX2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICBhc3NlcnQgY29waWVkX3NoYSA9PSBpbml0aWFsaXplclsicG9saWN5X3NoYTI1NiJdCiAgICBjaGVja3BvaW50ID0gYXJncy5vdXRwdXQgLyAicnVuIiAvIHJlc3VsdC5maW5hbF9jaGVja3BvaW50CiAgICBkYXRhID0gY2hlY2twb2ludC5yZWFkX2J5dGVzKCkKICAgIGFzc2VydCBsZW4oZGF0YSkgPT0gaW5pdGlhbGl6ZXJbInBhcmFtZXRlcl9jb3VudCJdICogNAogICAgYXNzZXJ0IGFsbChtYXRoLmlzZmluaXRlKHZhbHVlWzBdKSBmb3IgdmFsdWUgaW4gc3RydWN0Lml0ZXJfdW5wYWNrKCI8ZiIsIGRhdGEpKQogICAgcHJpbnQoIk5BVElWRV9GSVhFRF9QUklPUl9QSUxPVF9PSyIsIHJlc3VsdC50cmFpbmVkX3RpbWVzdGVwcywgaGFzaGxpYi5zaGEyNTYoZGF0YSkuaGV4ZGlnZXN0KCksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/native-fixed-prior75-gae99-h64-33m-v2-$job
mkdir -p "$out"
printf '%s  %s\n' 00bc98e5a0e082d083584aab43edcba2c68251c27706aff8fbab10a9b0953f01 "$disk/train_native_fixed_prior-gae99-20260927.py" | sha256sum -c -
python3 - "$disk/native-fixed-prior75-h64-advnorm-build-v1-27035/build/build.json" <<'PYINNER'
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
name=relh-classic-native-fixed-prior75-gae99-h64-33m-v2-$job
cleanup() { kill "${sampler:-}" 2>/dev/null || true; docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/native-fixed-prior75-gae99-h64-33m-v2-gpu-$job.csv" 2>&1 &
sampler=$!
ln -s "$out/run" "$disk/native-fixed-prior75-gae99-h64-33m-v2-pilot-$job-0"
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
    relh-generals-b300:20260923b python /recovery/train_native_fixed_prior-gae99-20260927.py \
      --template /work/normalized-clean-pilot-16788/config.json \
      --build /recovery/native-fixed-prior75-h64-advnorm-build-v1-27035/build \
      --output /recovery-output --learning-rate .003 --timesteps 33554432 --replay-ratio 4 --checkpoint-interval 8 --normalize-advantages --total-agents 16384 --horizon 64 --gae-lambda .99 > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval_v4.py" \
        "$disk" "$job" "$((now-start))" native-fixed-prior75-gae99-h64-33m-v2 600 1048576 30000; then
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

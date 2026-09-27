#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-teacher-agreement
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-teacher-agreement-%j.log
set -euo pipefail
export TMPDIR=/var/tmp/relh-generals-recovery
mkdir -p "$TMPDIR"
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'audit_native_raw_teacher_agreement-20260927.py'
data=base64.b64decode('IiIiQ29tcGFyZSByYXcgbmF0aXZlIFBQTyBjaGVja3BvaW50cyBvbiB0aGUgc2FtZSBsZWdhbCB0ZWFjaGVyIHRyYWplY3Rvcmllcy4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQgaGFzaGxpYgppbXBvcnQganNvbgpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBqYXgKaW1wb3J0IGpheC5udW1weSBhcyBqbnAKaW1wb3J0IG51bXB5IGFzIG5wCmZyb20gbWV0dGFfdHJhaW5pbmcuZW52aXJvbm1lbnQgaW1wb3J0IEVudmlyb25tZW50Q29udGV4dAoKZnJvbSBpbnRlZ3JhdGlvbnMubWV0dGFfcHVmZmVyIGltcG9ydCBCYXRjaGVkR2VuZXJhbHNQdWZmZXJFbnZpcm9ubWVudApmcm9tIGludGVncmF0aW9ucy5uYXRpdmVfcHVmZmVyX3BvbGljeSBpbXBvcnQgTmF0aXZlUHVmZmVyUG9saWN5CgoKZGVmIG1haW4oKSAtPiBOb25lOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tdHJhaW5pbmciLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWNoZWNrcG9pbnQiLCB0eXBlPVBhdGgsIGFjdGlvbj0iYXBwZW5kIiwgcmVxdWlyZWQ9VHJ1ZSkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tZ2FtZXMiLCB0eXBlPWludCwgZGVmYXVsdD0xMjgpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLXR1cm5zIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTI4KQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1zZWVkIiwgdHlwZT1pbnQsIGRlZmF1bHQ9MTM4NikKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBhc3NlcnQgamF4LmRldmljZXMoKVswXS5wbGF0Zm9ybSA9PSAiZ3B1IgogICAgYXNzZXJ0IGFyZ3MuZ2FtZXMgPiAwIGFuZCBhcmdzLnR1cm5zID4gMAogICAgbWFuaWZlc3QgPSBqc29uLmxvYWRzKGFyZ3MuYnVpbGQucmVhZF90ZXh0KCkpCiAgICBvcHRpb25zID0gbWFuaWZlc3RbImNvbmZpZyJdWyJweXRob25fZW52aXJvbm1lbnQiXVsib3B0aW9ucyJdLmNvcHkoKQogICAgYXNzZXJ0IG9wdGlvbnNbImNvd29ybGRfY2xhc3NpYyJdIGFuZCBub3Qgb3B0aW9uc1siaGludF9mZWF0dXJlcyJdCiAgICBvcHRpb25zLnVwZGF0ZSgKICAgICAgICBwYXJhbGxlbF9nYW1lcz1hcmdzLmdhbWVzLAogICAgICAgIG9wcG9uZW50PSJyYW5kb20iLAogICAgICAgIHRlYWNoZXI9ImV4cGFuZGVyX2hhcnZlc3RlciIsCiAgICAgICAgc3BhcnNlX3RlYWNoZXI9VHJ1ZSwKICAgICAgICBpbWl0YXRpb25fd2VpZ2h0PTAuMCwKICAgICAgICBzdXBlcnZpc2VfdGVhY2hlcj1GYWxzZSwKICAgICAgICB0ZWFjaGVyX3JvbGxvdXRzPUZhbHNlLAogICAgKQogICAgZW52ID0gQmF0Y2hlZEdlbmVyYWxzUHVmZmVyRW52aXJvbm1lbnQoCiAgICAgICAgY29udGV4dD1FbnZpcm9ubWVudENvbnRleHQoc2VlZD1hcmdzLnNlZWQsIGluZGV4PTAsIG1vZGU9InRyYWluIiwgb3V0cHV0PVBhdGgoIi90bXAiKSksCiAgICAgICAgKipvcHRpb25zLAogICAgKQogICAgY2hlY2twb2ludHMgPSBbXQogICAgZm9yIHBhdGggaW4gYXJncy5jaGVja3BvaW50OgogICAgICAgIGRpZ2VzdCA9IGhhc2hsaWIuc2hhMjU2KHBhdGgucmVhZF9ieXRlcygpKS5oZXhkaWdlc3QoKQogICAgICAgIHBvbGljeSA9IE5hdGl2ZVB1ZmZlclBvbGljeShhcmdzLmJ1aWxkLCBhcmdzLnRyYWluaW5nLCBwYXRoLCBkaWdlc3QpCiAgICAgICAgYXNzZXJ0IHBvbGljeS5sYXllcnMgPT0gMAogICAgICAgIGNoZWNrcG9pbnRzLmFwcGVuZCgocGF0aC5uYW1lLCBkaWdlc3QsIHBvbGljeSkpCiAgICBvYnNlcnZhdGlvbnMgPSBlbnYucmVzZXQoZiJ0ZWFjaGVyLWF1ZGl0LXthcmdzLnNlZWR9IikKICAgIHRvdGFscyA9IHsKICAgICAgICBuYW1lOiB7InNvdXJjZV90b3AxIjogMCwgImpvaW50X3RvcDEiOiAwLCAic291cmNlX3Byb2IiOiAwLjAsICJqb2ludF9wcm9iIjogMC4wfQogICAgICAgIGZvciBuYW1lLCBfLCBfIGluIGNoZWNrcG9pbnRzCiAgICB9CiAgICBjb3VudCA9IDAKICAgIHRyeToKICAgICAgICBmb3IgXyBpbiByYW5nZShhcmdzLnR1cm5zKToKICAgICAgICAgICAgbGFiZWxzID0gbnAuYXNhcnJheShvYnNlcnZhdGlvbnMucmVwbGF5X21ldGFkYXRhLCBkdHlwZT1ucC5pbnQzMikKICAgICAgICAgICAgYWN0aXZlID0gKH5lbnYuZmluaXNoZWQpICYgKGxhYmVsc1s6LCAwXSA+PSAwKQogICAgICAgICAgICBtYXNrcyA9IG5wLmFzYXJyYXkob2JzZXJ2YXRpb25zLmFjdGlvbl9tYXNrcywgZHR5cGU9Ym9vbCkKICAgICAgICAgICAgYXNzZXJ0IG1hc2tzW25wLmFyYW5nZShhcmdzLmdhbWVzKVthY3RpdmVdLCBsYWJlbHNbYWN0aXZlLCAwXV0uYWxsKCkKICAgICAgICAgICAgdmFsdWVzID0gam5wLmFzYXJyYXkob2JzZXJ2YXRpb25zLnZhbHVlcywgZHR5cGU9am5wLmZsb2F0MzIpCiAgICAgICAgICAgIG1hc2tlZCA9IGpucC5hc2FycmF5KG1hc2tzKQogICAgICAgICAgICByb3cgPSBucC5hcmFuZ2UoYXJncy5nYW1lcylbYWN0aXZlXQogICAgICAgICAgICBzb3VyY2UgPSBsYWJlbHNbYWN0aXZlLCAwXQogICAgICAgICAgICBzcGxpdCA9IG5wLndoZXJlKGxhYmVsc1thY3RpdmUsIDFdID49IDAsIGxhYmVsc1thY3RpdmUsIDFdLCAwKQogICAgICAgICAgICBmb3IgbmFtZSwgXywgcG9saWN5IGluIGNoZWNrcG9pbnRzOgogICAgICAgICAgICAgICAgbG9naXRzLCBfID0gcG9saWN5LmZvcndhcmQodmFsdWVzLCBwb2xpY3kuaW5pdGlhbF9zdGF0ZShhcmdzLmdhbWVzKSkKICAgICAgICAgICAgICAgIGxvZ2l0cyA9IGpucC53aGVyZShtYXNrZWQsIGxvZ2l0c1s6LCA6MTc2N10sIC1qbnAuaW5mKQogICAgICAgICAgICAgICAgc291cmNlX3Byb2JzID0gbnAuYXNhcnJheShqYXgubm4uc29mdG1heChsb2dpdHNbOiwgOjE3NjVdLCBheGlzPS0xKSkKICAgICAgICAgICAgICAgIHNwbGl0X3Byb2JzID0gbnAuYXNhcnJheShqYXgubm4uc29mdG1heChsb2dpdHNbOiwgMTc2NTpdLCBheGlzPS0xKSkKICAgICAgICAgICAgICAgIHNvdXJjZV9jaG9pY2UgPSBucC5hc2FycmF5KGpucC5hcmdtYXgobG9naXRzWzosIDoxNzY1XSwgYXhpcz0tMSkpCiAgICAgICAgICAgICAgICBzcGxpdF9jaG9pY2UgPSBucC5hc2FycmF5KGpucC5hcmdtYXgobG9naXRzWzosIDE3NjU6XSwgYXhpcz0tMSkpCiAgICAgICAgICAgICAgICBzZWxlY3RlZF9zb3VyY2VfcHJvYiA9IHNvdXJjZV9wcm9ic1tyb3csIHNvdXJjZV0KICAgICAgICAgICAgICAgIHNlbGVjdGVkX3NwbGl0X3Byb2IgPSBzcGxpdF9wcm9ic1tyb3csIHNwbGl0XQogICAgICAgICAgICAgICAgdG90YWwgPSB0b3RhbHNbbmFtZV0KICAgICAgICAgICAgICAgIHRvdGFsWyJzb3VyY2VfdG9wMSJdICs9IGludCgoc291cmNlX2Nob2ljZVthY3RpdmVdID09IHNvdXJjZSkuc3VtKCkpCiAgICAgICAgICAgICAgICB0b3RhbFsiam9pbnRfdG9wMSJdICs9IGludCgoKHNvdXJjZV9jaG9pY2VbYWN0aXZlXSA9PSBzb3VyY2UpICYgKHNwbGl0X2Nob2ljZVthY3RpdmVdID09IHNwbGl0KSkuc3VtKCkpCiAgICAgICAgICAgICAgICB0b3RhbFsic291cmNlX3Byb2IiXSArPSBmbG9hdChzZWxlY3RlZF9zb3VyY2VfcHJvYi5zdW0oKSkKICAgICAgICAgICAgICAgIHRvdGFsWyJqb2ludF9wcm9iIl0gKz0gZmxvYXQoKHNlbGVjdGVkX3NvdXJjZV9wcm9iICogc2VsZWN0ZWRfc3BsaXRfcHJvYikuc3VtKCkpCiAgICAgICAgICAgIGNvdW50ICs9IGxlbihyb3cpCiAgICAgICAgICAgIGNob3NlbiA9IG5wLndoZXJlKGxhYmVsc1s6LCAwXSA+PSAwLCBsYWJlbHNbOiwgMF0sIDE3NjQpCiAgICAgICAgICAgIHNwbGl0ID0gbnAud2hlcmUobGFiZWxzWzosIDFdID49IDAsIGxhYmVsc1s6LCAxXSwgMCkKICAgICAgICAgICAgb2JzZXJ2YXRpb25zID0gZW52LnN0ZXAobnAuc3RhY2soKGNob3Nlbiwgc3BsaXQpLCBheGlzPTEpKS5vYnNlcnZhdGlvbgogICAgZmluYWxseToKICAgICAgICBlbnYuY2xvc2UoKQogICAgYXNzZXJ0IGNvdW50ID4gMAogICAgcHJpbnQoanNvbi5kdW1wcyh7CiAgICAgICAgInNjb3BlIjogIlNhbWUgdGVhY2hlci1kcml2ZW4gQ2xhc3NpYyBzdGF0ZXM7IHJhdyBuYXRpdmUgYWN0b3I7IGRpYWdub3N0aWMgb25seSIsCiAgICAgICAgImdhbWVzIjogYXJncy5nYW1lcywKICAgICAgICAidHVybnMiOiBhcmdzLnR1cm5zLAogICAgICAgICJsYWJlbGVkX2RlY2lzaW9ucyI6IGNvdW50LAogICAgICAgICJwb2xpY2llcyI6IHsKICAgICAgICAgICAgbmFtZTogeyJzaGEyNTYiOiBkaWdlc3QsICoqe2tleTogdmFsdWUgLyBjb3VudCBmb3Iga2V5LCB2YWx1ZSBpbiB0b3RhbHNbbmFtZV0uaXRlbXMoKX19CiAgICAgICAgICAgIGZvciBuYW1lLCBkaWdlc3QsIF8gaW4gY2hlY2twb2ludHMKICAgICAgICB9LAogICAgfSksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
if path.exists(): assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
printf '%s  %s\n' bab162d6088f2acad9e3a4d57c51629def931ee942b18f1e903b31474ac40e77 "$disk/audit_native_raw_teacher_agreement-20260927.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-native-raw-teacher-agreement-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
timeout -k 30s 10m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" \
    -v "$disk/native-hint-prior-puffer.py-v2-20260927:/work/metta-device-wide-source/metta_training/puffer.py:ro" \
    -v "$disk/native-hint-prior-model_config.py-v2-20260927:/work/metta-device-wide-source/metta_training/model_config.py:ro" \
    -v "$disk/native-advnorm-native_build-v2-20260927.py:/work/metta-device-wide-source/metta_training/native_build.py:ro" \
    -v "$disk/native-advnorm-learner-python-v1-20260927.py:/work/metta-device-wide-source/metta_training/learner.py:ro" \
    -v "$disk/native-advnorm-advantage.cuh-20260927:/work/metta-device-wide-source/metta_training/native/advantage.cuh:ro" \
    -v "$disk/native-advnorm-learner-v3-20260927.cuh:/work/metta-device-wide-source/metta_training/native/learner.cuh:ro" \
    -v "$disk/native-advnorm-replay-v2-20260927.cuh:/work/metta-device-wide-source/metta_training/native/replay.cuh:ro" \
    -v "$disk/native_puffer_policy-fixed-prior-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v "$disk/metta_puffer-terminal-boundary-fixed-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery/cache-native-raw-audit \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/audit_native_raw_teacher_agreement-20260927.py \
      --build /recovery/native-raw-imitation-diag-build-v1-27054/build/build.json \
      --training /recovery/native-raw-imitation-diag-r4-8m-27059/run/training.json \
      --checkpoint /recovery/native-raw-imitation-diag-r4-8m-27059/run/checkpoints/metta_generals/run/0000000002097152.bin \
      --checkpoint /recovery/native-raw-imitation-diag-r4-8m-27059/run/checkpoints/metta_generals/run/0000000008388608.bin \
      --games 128 --turns 128 --seed 1386
NODE

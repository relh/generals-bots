#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-own-action-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-own-action-audit-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64, sys
from pathlib import Path
path = Path(sys.argv[1]) / 'audit_native_raw_actions-20260927.py'
data = base64.b64decode('IiIiQXVkaXQgbmF0aXZlIGdyZWVkeSBhY3Rpb25zIG9uIGVhY2ggcmF3IHBvbGljeSdzIG93biBDbGFzc2ljIHRyYWplY3Rvcmllcy4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQganNvbgpmcm9tIGNvbGxlY3Rpb25zIGltcG9ydCBDb3VudGVyCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKaW1wb3J0IGpheAppbXBvcnQgamF4Lm51bXB5IGFzIGpucAppbXBvcnQgbnVtcHkgYXMgbnAKZnJvbSBtZXR0YV90cmFpbmluZy5lbnZpcm9ubWVudCBpbXBvcnQgRW52aXJvbm1lbnRDb250ZXh0Cgpmcm9tIGludGVncmF0aW9ucy5tZXR0YV9wdWZmZXIgaW1wb3J0IEJhdGNoZWRHZW5lcmFsc1B1ZmZlckVudmlyb25tZW50CmZyb20gaW50ZWdyYXRpb25zLm5hdGl2ZV9wdWZmZXJfcG9saWN5IGltcG9ydCBOYXRpdmVQdWZmZXJQb2xpY3kKCgpkZWYgbWFpbigpOgogICAgcGFyc2VyID0gYXJncGFyc2UuQXJndW1lbnRQYXJzZXIoKQogICAgcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1jb25maWciLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLW91dHB1dCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBjb25maWcgPSBqc29uLmxvYWRzKGFyZ3MuY29uZmlnLnJlYWRfdGV4dCgpKQogICAgYXJncy5vdXRwdXQubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1GYWxzZSkKICAgIG9wdGlvbnMgPSBqc29uLmxvYWRzKFBhdGgoY29uZmlnWyJlbnZpcm9ubWVudF9idWlsZCJdKS5yZWFkX3RleHQoKSlbImNvbmZpZyJdWyJweXRob25fZW52aXJvbm1lbnQiXVsib3B0aW9ucyJdCiAgICBhc3NlcnQgb3B0aW9uc1siY293b3JsZF9jbGFzc2ljIl0gYW5kIG5vdCBvcHRpb25zWyJoaW50X2ZlYXR1cmVzIl0KICAgIG9wdGlvbnMudXBkYXRlKAogICAgICAgIHBhcmFsbGVsX2dhbWVzPWNvbmZpZ1siZ2FtZXMiXSwgb3Bwb25lbnQ9InN0cm9uZ19taXhlZCIsIHN1cGVydmlzZV90ZWFjaGVyPUZhbHNlLCB0ZWFjaGVyX3JvbGxvdXRzPUZhbHNlCiAgICApCiAgICByZXN1bHQgPSBkaWN0KAogICAgICAgIHNjb3BlPSJPd24gZ3JlZWR5IHRyYWplY3RvcmllcyBvbiByYXcgcHVibGljIENsYXNzaWMgb2JzZXJ2YXRpb25zOyBkaWFnbm9zdGljLCBub3QgaGVsZC1vdXQgd2lucyIsCiAgICAgICAgY29uZmlnPWNvbmZpZywKICAgICAgICBwb2xpY2llcz17fSwKICAgICkKICAgIGZvciBuYW1lLCByZWZlcmVuY2UgaW4gY29uZmlnWyJwb2xpY2llcyJdLml0ZW1zKCk6CiAgICAgICAgcG9saWN5ID0gTmF0aXZlUHVmZmVyUG9saWN5KAogICAgICAgICAgICAqKntrZXk6IHZhbHVlIGlmIGtleSA9PSAic2hhMjU2IiBlbHNlIFBhdGgodmFsdWUpIGZvciBrZXksIHZhbHVlIGluIHJlZmVyZW5jZS5pdGVtcygpfQogICAgICAgICkKICAgICAgICBvdXRwdXQgPSBhcmdzLm91dHB1dCAvIG5hbWUKICAgICAgICBvdXRwdXQubWtkaXIoKQogICAgICAgIGVudiA9IEJhdGNoZWRHZW5lcmFsc1B1ZmZlckVudmlyb25tZW50KAogICAgICAgICAgICBjb250ZXh0PUVudmlyb25tZW50Q29udGV4dChzZWVkPWNvbmZpZ1sic2VlZCJdLCBpbmRleD0wLCBtb2RlPSJldmFsdWF0ZSIsIG91dHB1dD1vdXRwdXQpLCAqKm9wdGlvbnMKICAgICAgICApCiAgICAgICAgc3RhdGUgPSBwb2xpY3kuaW5pdGlhbF9zdGF0ZShjb25maWdbImdhbWVzIl0pCiAgICAgICAgb2JzZXJ2YXRpb24gPSBlbnYucmVzZXQoc3RyKGNvbmZpZ1sic2VlZCJdKSkKICAgICAgICB3aW5kb3dzID0gW10KICAgICAgICB0cnk6CiAgICAgICAgICAgIGZvciBzdGFydCBpbiByYW5nZSgwLCBjb25maWdbInR1cm5zIl0sIDUwKToKICAgICAgICAgICAgICAgIGNvdW50cyA9IENvdW50ZXIoKQogICAgICAgICAgICAgICAgbW92ZV9jb3VudHMgPSBDb3VudGVyKCkKICAgICAgICAgICAgICAgIGZvciB0dXJuIGluIHJhbmdlKHN0YXJ0LCBtaW4oc3RhcnQgKyA1MCwgY29uZmlnWyJ0dXJucyJdKSk6CiAgICAgICAgICAgICAgICAgICAgYWN0aXZlID0gfmVudi5maW5pc2hlZC5jb3B5KCkKICAgICAgICAgICAgICAgICAgICBpZiBub3QgYWN0aXZlLmFueSgpOgogICAgICAgICAgICAgICAgICAgICAgICBicmVhawogICAgICAgICAgICAgICAgICAgIHZhbHVlcyA9IG5wLmFzYXJyYXkob2JzZXJ2YXRpb24udmFsdWVzLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgICAgICAgICAgICAgICAgIG1hc2tzID0gbnAuYXNhcnJheShvYnNlcnZhdGlvbi5hY3Rpb25fbWFza3MsIGR0eXBlPWJvb2wpCiAgICAgICAgICAgICAgICAgICAgZGVjb2RlZCwgc3RhdGUgPSBwb2xpY3kuZm9yd2FyZChqbnAuYXNhcnJheSh2YWx1ZXMpLCBzdGF0ZSkKICAgICAgICAgICAgICAgICAgICBsb2dpdHMgPSBqbnAud2hlcmUoam5wLmFzYXJyYXkobWFza3MpLCBkZWNvZGVkWzosIDoxNzY3XSwgLWpucC5pbmYpCiAgICAgICAgICAgICAgICAgICAgbW92ZXMgPSBqYXgubm4uc29mdG1heChsb2dpdHNbOiwgOjE3NjVdLCBheGlzPS0xKQogICAgICAgICAgICAgICAgICAgIGNob2ljZXMgPSBucC5hc2FycmF5KAogICAgICAgICAgICAgICAgICAgICAgICBqbnAuc3RhY2soCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAoam5wLmFyZ21heChsb2dpdHNbOiwgOjE3NjVdLCBheGlzPS0xKSwgam5wLmFyZ21heChsb2dpdHNbOiwgMTc2NTpdLCBheGlzPS0xKSksIGF4aXM9LTEKICAgICAgICAgICAgICAgICAgICAgICAgKSwKICAgICAgICAgICAgICAgICAgICAgICAgZHR5cGU9bnAuaW50MzIsCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgICAgIHByb2JhYmlsaXRpZXMgPSBucC5hc2FycmF5KG1vdmVzKQogICAgICAgICAgICAgICAgICAgIGFzc2VydCBtYXNrc1tucC5hcmFuZ2UoY29uZmlnWyJnYW1lcyJdKSwgY2hvaWNlc1s6LCAwXV0uYWxsKCkKICAgICAgICAgICAgICAgICAgICBhc3NlcnQgbWFza3NbbnAuYXJhbmdlKGNvbmZpZ1siZ2FtZXMiXSksIDE3NjUgKyBjaG9pY2VzWzosIDFdXS5hbGwoKQogICAgICAgICAgICAgICAgICAgIGxlZ2FsX21vdmVzID0gbWFza3NbOiwgOjE3NjRdLnN1bShheGlzPTEpCiAgICAgICAgICAgICAgICAgICAgbW92ZV9lbnRyb3B5ID0gLShwcm9iYWJpbGl0aWVzICogbnAubG9nKG5wLm1heGltdW0ocHJvYmFiaWxpdGllcywgMWUtMzApKSkuc3VtKGF4aXM9MSkKICAgICAgICAgICAgICAgICAgICBjb3VudHNbImRlY2lzaW9ucyJdICs9IGludChhY3RpdmUuc3VtKCkpCiAgICAgICAgICAgICAgICAgICAgY291bnRzWyJwYXNzZXMiXSArPSBpbnQoKGFjdGl2ZSAmIChjaG9pY2VzWzosIDBdID09IDE3NjQpKS5zdW0oKSkKICAgICAgICAgICAgICAgICAgICBjb3VudHNbInBhc3Nlc193aXRoX21vdmVzIl0gKz0gaW50KChhY3RpdmUgJiAobGVnYWxfbW92ZXMgPiAwKSAmIChjaG9pY2VzWzosIDBdID09IDE3NjQpKS5zdW0oKSkKICAgICAgICAgICAgICAgICAgICBjb3VudHNbIm1vdmFibGUiXSArPSBpbnQoKGFjdGl2ZSAmIChsZWdhbF9tb3ZlcyA+IDApKS5zdW0oKSkKICAgICAgICAgICAgICAgICAgICBjb3VudHNbInNwbGl0X29uZSJdICs9IGludCgoYWN0aXZlICYgKGNob2ljZXNbOiwgMV0gPT0gMSkpLnN1bSgpKQogICAgICAgICAgICAgICAgICAgIGNvdW50c1sibGVnYWxfbW92ZXMiXSArPSBpbnQobGVnYWxfbW92ZXNbYWN0aXZlXS5zdW0oKSkKICAgICAgICAgICAgICAgICAgICBjb3VudHNbImVudHJvcHlfbWlsbGkiXSArPSBpbnQobnAucmludChtb3ZlX2VudHJvcHlbYWN0aXZlXS5zdW0oKSAqIDEwMDApKQogICAgICAgICAgICAgICAgICAgIGNvdW50c1sibWF4X3Byb2JhYmlsaXR5X21pbGxpIl0gKz0gaW50KG5wLnJpbnQocHJvYmFiaWxpdGllc1thY3RpdmVdLm1heChheGlzPTEpLnN1bSgpICogMTAwMCkpCiAgICAgICAgICAgICAgICAgICAgbW92ZV9jb3VudHMudXBkYXRlKG1hcChpbnQsIGNob2ljZXNbYWN0aXZlLCAwXSkpCiAgICAgICAgICAgICAgICAgICAgb2JzZXJ2YXRpb24gPSBlbnYuc3RlcChjaG9pY2VzKS5vYnNlcnZhdGlvbgogICAgICAgICAgICAgICAgICAgIGlmIHBvbGljeS5sYXllcnM6CiAgICAgICAgICAgICAgICAgICAgICAgIGRvbmUgPSBucC5hc2FycmF5KGVudi5maW5pc2hlZCkKICAgICAgICAgICAgICAgICAgICAgICAgc3RhdGUgPSBqbnAud2hlcmUoam5wLmFzYXJyYXkoZG9uZSlbTm9uZSwgOiwgTm9uZV0sIDAsIHN0YXRlKQogICAgICAgICAgICAgICAgaWYgbm90IGNvdW50c1siZGVjaXNpb25zIl06CiAgICAgICAgICAgICAgICAgICAgYnJlYWsKICAgICAgICAgICAgICAgIHRvdGFsID0gY291bnRzWyJkZWNpc2lvbnMiXQogICAgICAgICAgICAgICAgd2luZG93cy5hcHBlbmQoCiAgICAgICAgICAgICAgICAgICAgZGljdCgKICAgICAgICAgICAgICAgICAgICAgICAgZmlyc3RfdHVybj1zdGFydCwKICAgICAgICAgICAgICAgICAgICAgICAgbGFzdF90dXJuPW1pbihzdGFydCArIDQ5LCBjb25maWdbInR1cm5zIl0gLSAxKSwKICAgICAgICAgICAgICAgICAgICAgICAgZGVjaXNpb25zPXRvdGFsLAogICAgICAgICAgICAgICAgICAgICAgICBwYXNzX2ZyYWN0aW9uPWNvdW50c1sicGFzc2VzIl0gLyB0b3RhbCwKICAgICAgICAgICAgICAgICAgICAgICAgcGFzc193aXRoX21vdmVzX2ZyYWN0aW9uPSgKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNvdW50c1sicGFzc2VzX3dpdGhfbW92ZXMiXSAvIGNvdW50c1sibW92YWJsZSJdIGlmIGNvdW50c1sibW92YWJsZSJdIGVsc2UgMAogICAgICAgICAgICAgICAgICAgICAgICApLAogICAgICAgICAgICAgICAgICAgICAgICBzcGxpdF9vbmVfZnJhY3Rpb249Y291bnRzWyJzcGxpdF9vbmUiXSAvIHRvdGFsLAogICAgICAgICAgICAgICAgICAgICAgICBsZWdhbF9tb3Zlc19tZWFuPWNvdW50c1sibGVnYWxfbW92ZXMiXSAvIHRvdGFsLAogICAgICAgICAgICAgICAgICAgICAgICBtb3ZlX2VudHJvcHk9Y291bnRzWyJlbnRyb3B5X21pbGxpIl0gLyAoMTAwMCAqIHRvdGFsKSwKICAgICAgICAgICAgICAgICAgICAgICAgbW92ZV9tYXhfcHJvYmFiaWxpdHk9Y291bnRzWyJtYXhfcHJvYmFiaWxpdHlfbWlsbGkiXSAvICgxMDAwICogdG90YWwpLAogICAgICAgICAgICAgICAgICAgICAgICB0b3BfbW92ZXM9bW92ZV9jb3VudHMubW9zdF9jb21tb24oNSksCiAgICAgICAgICAgICAgICAgICAgKQogICAgICAgICAgICAgICAgKQogICAgICAgIGZpbmFsbHk6CiAgICAgICAgICAgIGVudi5jbG9zZSgpCiAgICAgICAgcmVzdWx0WyJwb2xpY2llcyJdW25hbWVdID0gZGljdChjaGVja3BvaW50X3NoYTI1Nj1yZWZlcmVuY2VbInNoYTI1NiJdLCB3aW5kb3dzPXdpbmRvd3MpCiAgICAoYXJncy5vdXRwdXQgLyAiYXVkaXQuanNvbiIpLndyaXRlX3RleHQoanNvbi5kdW1wcyhyZXN1bHQsIGluZGVudD0yKSArICJcbiIpCiAgICBwcmludChqc29uLmR1bXBzKHJlc3VsdFsicG9saWNpZXMiXSksIGZsdXNoPVRydWUpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
if path.exists():
    assert path.read_bytes() == data
else:
    with path.open('xb') as handle:
        handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/native-raw-own-action-audit-$SLURM_JOB_ID
mkdir -p "$out"
printf '%s  %s\n' d47023ca9f7e42b505866b7bf79dc6df83f4bd3e47a16237da75c424de555b28 \
    "$disk/audit_native_raw_actions-20260927.py" | sha256sum -c -
printf '%s  %s\n' f3a5b281a1b648c1f93db9e0033ceab578207f74e26f521b8e159253df265f9a \
    "$disk/native_puffer_policy-fixed-prior-20260927.py" | sha256sum -c -
python3 - "$disk" "$out/config.json" <<'PYCONFIG'
import hashlib
import json
import sys
from pathlib import Path

disk, output = map(Path, sys.argv[1:])
root = Path('/recovery')
raw_build = root / 'native-raw-ppo-build-26769/build/build.json'
entries = {
    'raw_initial': ('native-raw-ppo-134m-26780/run',
                    'checkpoints/metta_generals/run/0000000016777216.bin', raw_build),
    'raw_constant': ('native-raw-ppo-r4-noanneal-33m-26836/run',
                     'checkpoints/metta_generals/run/0000000033554432.bin', raw_build),
}
policies = {}
for name, (run, checkpoint, build) in entries.items():
    local = disk / run / checkpoint
    assert local.is_file()
    policies[name] = dict(
        build=str(build), training=str(root / run / 'training.json'),
        checkpoint=str(root / run / checkpoint),
        sha256=hashlib.sha256(local.read_bytes()).hexdigest(),
    )
output.write_text(json.dumps(dict(
    policies=policies, environment_build=str(raw_build),
    seed=1390, games=64, turns=192, parent='raw_initial',
), indent=2) + '\n')
print(json.dumps({name: ref['sha256'] for name, ref in policies.items()}), flush=True)
PYCONFIG
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-native-raw-own-action-audit-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
timeout -k 30s 10m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output" \
    -v "$disk/native_puffer_policy-fixed-prior-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v "$disk/metta_puffer-terminal-boundary-fixed-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/audit_native_raw_actions-20260927.py \
    --config /recovery-output/config.json --output /recovery-output/proof > "$out/container.log" 2>&1
cat "$out/proof/audit.json"
NODE

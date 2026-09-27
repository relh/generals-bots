#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-hint-prior-parity
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-hint-prior-parity-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'native_puffer_policy-fixed-prior-20260927.py'
data=base64.b64decode('IiIiRnJvemVuIGZsb2F0MzIgTWluR1JVIGluZmVyZW5jZSBmb3IgdGhlIHBpbm5lZCBkZWZhdWx0IFB1ZmZlcjUgYXJjaGl0ZWN0dXJlLgoKTWF0Y2hlcyBlbmNvZGVyL2RlY29kZXIgYW5kIG1pbmdydV9nYXRlIGluIHVwc3RyZWFtIDZmZmE1YjEwIHNyYy9hbGdvLmN1LgpUaGlzIGRlbGliZXJhdGVseSBhY2NlcHRzIG9ubHkgdGhlIHZlcmlmaWVkIENsYXNzaWMgbmF0aXZlIGJ1aWxkIGNvbnRyYWN0LgoiIiIKCmltcG9ydCBoYXNobGliCmltcG9ydCBqc29uCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKaW1wb3J0IGpheAppbXBvcnQgamF4Lm51bXB5IGFzIGpucAppbXBvcnQgbnVtcHkgYXMgbnAKCgpjbGFzcyBOYXRpdmVQdWZmZXJQb2xpY3k6CiAgICBkZWYgX19pbml0X18oc2VsZiwgYnVpbGQ6IFBhdGgsIHRyYWluaW5nOiBQYXRoLCBjaGVja3BvaW50OiBQYXRoLCBzaGEyNTY6IHN0cik6CiAgICAgICAgbWFuaWZlc3QgPSBqc29uLmxvYWRzKGJ1aWxkLnJlYWRfdGV4dCgpKQogICAgICAgIHJlY29yZCA9IGpzb24ubG9hZHModHJhaW5pbmcucmVhZF90ZXh0KCkpCiAgICAgICAgaWYgbWFuaWZlc3QgIT0gcmVjb3JkWyJidWlsZCJdOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJUcmFpbmluZy9idWlsZCBtYW5pZmVzdHMgZGlmZmVyIikKICAgICAgICBjb25maWcgPSBtYW5pZmVzdFsiY29uZmlnIl0KICAgICAgICBzcGVjID0gY29uZmlnWyJweXRob25fZW52aXJvbm1lbnQiXVsic3BlYyJdCiAgICAgICAgaWYgKG1hbmlmZXN0WyJyZXZpc2lvbiJdICE9ICI2ZmZhNWIxMGRiYmJlNGQxZTgyODgzNjdjN2Q5ZDNhY2QzYmFkNGEyIgogICAgICAgICAgICAgICAgb3IgY29uZmlnWyJmYWJyaWMiXSBpcyBub3QgTm9uZSBvciBjb25maWdbInByZWNpc2lvbiJdICE9ICJmbG9hdDMyIgogICAgICAgICAgICAgICAgb3Igc3BlY1sib2JzZXJ2YXRpb25fc2l6ZSJdICE9IDYxNzQgb3Igc3BlY1siYWN0aW9uX3NpemVzIl0gIT0gWzE3NjUsIDJdCiAgICAgICAgICAgICAgICBvciBzcGVjWyJ0ZWFjaGVyIl0gb3Igc3BlY1sicm91dGluZyJdIG9yIHNwZWNbInJlcGxheV9tZXRhZGF0YV9zaXplIl0pOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbnN1cHBvcnRlZCBuYXRpdmUgcG9saWN5IGNvbnRyYWN0IikKICAgICAgICBzZWxmLmhpZGRlbiA9IGludChyZWNvcmRbImNvbmZpZyJdWyJvdmVycmlkZXMiXVsicG9saWN5LmhpZGRlbl9zaXplIl0pCiAgICAgICAgc2VsZi5sYXllcnMgPSBpbnQocmVjb3JkWyJjb25maWciXVsib3ZlcnJpZGVzIl1bInBvbGljeS5udW1fbGF5ZXJzIl0pCiAgICAgICAgc2VsZi5oaW50X3ByaW9yID0gY29uZmlnLmdldCgibmF0aXZlX2hpbnRfcHJpb3IiKQogICAgICAgIGlmIHNlbGYuaGludF9wcmlvciBpcyBub3QgTm9uZToKICAgICAgICAgICAgcHJpb3IgPSBzZWxmLmhpbnRfcHJpb3IKICAgICAgICAgICAgaWYgKHByaW9yWyJjaGFubmVscyJdICogcHJpb3JbImNlbGxzIl0gIT0gNjE3NCBvciBwcmlvclsiY2VsbHMiXSAhPSA0NDEKICAgICAgICAgICAgICAgICAgICBvciBwcmlvclsibW92ZV9jaGFubmVsIl0gKyA0ID4gcHJpb3JbImNoYW5uZWxzIl0KICAgICAgICAgICAgICAgICAgICBvciBtYXgocHJpb3JbInBhc3NfY2hhbm5lbCJdLCBwcmlvclsic3BsaXRfY2hhbm5lbCJdKSA+PSBwcmlvclsiY2hhbm5lbHMiXSk6CiAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbnN1cHBvcnRlZCBuYXRpdmUgaGludCBwcmlvciIpCiAgICAgICAgaWYgc2VsZi5oaWRkZW4gPD0gMCBvciBzZWxmLmxheWVycyA8IDA6CiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoIkludmFsaWQgcmVjdXJyZW50IGRpbWVuc2lvbnMiKQogICAgICAgIGRhdGEgPSBjaGVja3BvaW50LnJlYWRfYnl0ZXMoKQogICAgICAgIGlmIGhhc2hsaWIuc2hhMjU2KGRhdGEpLmhleGRpZ2VzdCgpICE9IHNoYTI1NjoKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiQ2hlY2twb2ludCBTSEEyNTYgbWlzbWF0Y2giKQogICAgICAgIG9mZnNldCA9IDAKCiAgICAgICAgZGVmIHRha2Uoc2hhcGUpOgogICAgICAgICAgICBub25sb2NhbCBvZmZzZXQKICAgICAgICAgICAgIyBQdWZmZXIgQWxsb2NhdG9yIGFsaWducyB0aGUgc3RhcnQgb2YgZWFjaCByZWdpc3RlcmVkIHRlbnNvciB0bzE2IGJ5dGVzLgogICAgICAgICAgICBvZmZzZXQgPSAob2Zmc2V0ICsgMTUpICYgfjE1CiAgICAgICAgICAgIGNvdW50ID0gaW50KG5wLnByb2Qoc2hhcGUpKQogICAgICAgICAgICBhcnJheSA9IG5wLmZyb21idWZmZXIoZGF0YSwgZHR5cGU9IjxmNCIsIGNvdW50PWNvdW50LCBvZmZzZXQ9b2Zmc2V0KS5yZXNoYXBlKHNoYXBlKQogICAgICAgICAgICBvZmZzZXQgKz0gY291bnQgKiA0CiAgICAgICAgICAgIGlmIG5vdCBucC5pc2Zpbml0ZShhcnJheSkuYWxsKCk6CiAgICAgICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJOb25maW5pdGUgY2hlY2twb2ludCBwYXJhbWV0ZXJzIikKICAgICAgICAgICAgcmV0dXJuIGpucC5hc2FycmF5KGFycmF5KQoKICAgICAgICBzZWxmLmVuY29kZXIgPSB0YWtlKChzZWxmLmhpZGRlbiwgNjE3NCkpCiAgICAgICAgc2VsZi5kZWNvZGVyID0gdGFrZSgoMTc2OCwgc2VsZi5oaWRkZW4pKQogICAgICAgIHNlbGYucmVjdXJyZW50ID0gdHVwbGUodGFrZSgoMyAqIHNlbGYuaGlkZGVuLCBzZWxmLmhpZGRlbikpIGZvciBfIGluIHJhbmdlKHNlbGYubGF5ZXJzKSkKICAgICAgICBpZiBvZmZzZXQgIT0gbGVuKGRhdGEpOgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJVbmV4cGVjdGVkIGNoZWNrcG9pbnQgbGVuZ3RoIikKICAgICAgICBzZWxmLmZvcndhcmQgPSBqYXguaml0KHNlbGYuX2ZvcndhcmQpCgogICAgZGVmIGluaXRpYWxfc3RhdGUoc2VsZiwgbGFuZXMpOgogICAgICAgIHJldHVybiBqbnAuemVyb3MoKHNlbGYubGF5ZXJzLCBsYW5lcywgc2VsZi5oaWRkZW4pLCBkdHlwZT1qbnAuZmxvYXQzMikKCiAgICBkZWYgX2ZvcndhcmQoc2VsZiwgb2JzZXJ2YXRpb25zLCBzdGF0ZSk6CiAgICAgICAgeCA9IGpucC5tYXRtdWwob2JzZXJ2YXRpb25zLCBzZWxmLmVuY29kZXIuVCwgcHJlY2lzaW9uPWpheC5sYXguUHJlY2lzaW9uLkhJR0hFU1QpCiAgICAgICAgc3RhdGVzID0gW10KICAgICAgICBmb3IgbGF5ZXIsIHdlaWdodCBpbiBlbnVtZXJhdGUoc2VsZi5yZWN1cnJlbnQpOgogICAgICAgICAgICBjb21iaW5lZCA9IGpucC5tYXRtdWwoeCwgd2VpZ2h0LlQsIHByZWNpc2lvbj1qYXgubGF4LlByZWNpc2lvbi5ISUdIRVNUKQogICAgICAgICAgICBoaWRkZW4sIGdhdGUsIHByb2plY3Rpb24gPSBqbnAuc3BsaXQoY29tYmluZWQsIDMsIGF4aXM9LTEpCiAgICAgICAgICAgIHogPSBqYXgubm4uc2lnbW9pZChnYXRlKQogICAgICAgICAgICBjYW5kaWRhdGUgPSBqbnAud2hlcmUoaGlkZGVuID49IDAsIGhpZGRlbiArIC41LCBqYXgubm4uc2lnbW9pZChoaWRkZW4pKQogICAgICAgICAgICBkaWZmZXJlbmNlID0gY2FuZGlkYXRlIC0gc3RhdGVbbGF5ZXJdCiAgICAgICAgICAgICMgTWF0Y2ggbmF0aXZlIGxlcnAncyBzdGFibGUgYnJhbmNoZXMuCiAgICAgICAgICAgIG5leHRfc3RhdGUgPSBqbnAud2hlcmUoeiA8IC41LCBzdGF0ZVtsYXllcl0gKyB6ICogZGlmZmVyZW5jZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjYW5kaWRhdGUgLSBkaWZmZXJlbmNlICogKDEgLSB6KSkKICAgICAgICAgICAgcyA9IGpheC5ubi5zaWdtb2lkKHByb2plY3Rpb24pCiAgICAgICAgICAgIHggPSBzICogbmV4dF9zdGF0ZSArICgxIC0gcykgKiB4CiAgICAgICAgICAgIHN0YXRlcy5hcHBlbmQobmV4dF9zdGF0ZSkKICAgICAgICBkZWNvZGVkID0gam5wLm1hdG11bCh4LCBzZWxmLmRlY29kZXIuVCwgcHJlY2lzaW9uPWpheC5sYXguUHJlY2lzaW9uLkhJR0hFU1QpCiAgICAgICAgaWYgc2VsZi5oaW50X3ByaW9yIGlzIG5vdCBOb25lOgogICAgICAgICAgICBwcmlvciA9IHNlbGYuaGludF9wcmlvcgogICAgICAgICAgICBwbGFuZXMgPSBvYnNlcnZhdGlvbnMucmVzaGFwZSgtMSwgcHJpb3JbImNoYW5uZWxzIl0sIHByaW9yWyJjZWxscyJdKQogICAgICAgICAgICBtb3ZlcyA9IHByaW9yWyJtb3ZlX3NjYWxlIl0gKiBwbGFuZXNbOiwgcHJpb3JbIm1vdmVfY2hhbm5lbCJdOnByaW9yWyJtb3ZlX2NoYW5uZWwiXSArIDRdLnJlc2hhcGUoLTEsIDE3NjQpCiAgICAgICAgICAgIHBhc3NpbmcgPSBwcmlvclsicGFzc19zY2FsZSJdICogcGxhbmVzWzosIHByaW9yWyJwYXNzX2NoYW5uZWwiXSwgMF0KICAgICAgICAgICAgc3BsaXR0aW5nID0gcHJpb3JbInNwbGl0X3NjYWxlIl0gKiBwbGFuZXNbOiwgcHJpb3JbInNwbGl0X2NoYW5uZWwiXSwgMF0KICAgICAgICAgICAgZGVjb2RlZCA9IGRlY29kZWQgKyBqbnAuY29uY2F0ZW5hdGUoKAogICAgICAgICAgICAgICAgbW92ZXMsIHBhc3NpbmdbOiwgTm9uZV0sIC1zcGxpdHRpbmdbOiwgTm9uZV0sIHNwbGl0dGluZ1s6LCBOb25lXSwKICAgICAgICAgICAgICAgIGpucC56ZXJvcygob2JzZXJ2YXRpb25zLnNoYXBlWzBdLCAxKSwgZHR5cGU9ZGVjb2RlZC5kdHlwZSksCiAgICAgICAgICAgICksIGF4aXM9MSkKICAgICAgICByZXR1cm4gZGVjb2RlZCwgKAogICAgICAgICAgICBqbnAuc3RhY2soc3RhdGVzKSBpZiBzZWxmLmxheWVycyBlbHNlIHN0YXRlCiAgICAgICAgKQoKICAgIGRlZiBhY3Rpb25zKHNlbGYsIG9ic2VydmF0aW9ucywgbWFza3MsIHN0YXRlLCBrZXk9Tm9uZSk6CiAgICAgICAgaWYgb2JzZXJ2YXRpb25zLnNoYXBlICE9IChzdGF0ZS5zaGFwZVsxXSwgNjE3NCkgb3IgbWFza3Muc2hhcGUgIT0gKHN0YXRlLnNoYXBlWzFdLCAxNzY3KToKICAgICAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigiVW5leHBlY3RlZCBvYnNlcnZhdGlvbi9tYXNrIGRpbWVuc2lvbnMiKQogICAgICAgIG91dHB1dCwgc3RhdGUgPSBzZWxmLmZvcndhcmQoam5wLmFzYXJyYXkob2JzZXJ2YXRpb25zLCBkdHlwZT1qbnAuZmxvYXQzMiksIHN0YXRlKQogICAgICAgIGxvZ2l0cyA9IGpucC53aGVyZShqbnAuYXNhcnJheShtYXNrcywgZHR5cGU9Ym9vbCksIG91dHB1dFs6LCA6MTc2N10sIC1qbnAuaW5mKQogICAgICAgIGlmIGtleSBpcyBOb25lOgogICAgICAgICAgICBhY3Rpb25zID0gam5wLnN0YWNrKChqbnAuYXJnbWF4KGxvZ2l0c1s6LCA6MTc2NV0sIGF4aXM9LTEpLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBqbnAuYXJnbWF4KGxvZ2l0c1s6LCAxNzY1Ol0sIGF4aXM9LTEpKSwgYXhpcz0tMSkKICAgICAgICBlbHNlOgogICAgICAgICAgICBtb3ZlX2tleSwgc3BsaXRfa2V5ID0gamF4LnJhbmRvbS5zcGxpdChrZXkpCiAgICAgICAgICAgIGFjdGlvbnMgPSBqbnAuc3RhY2soKGpheC5yYW5kb20uY2F0ZWdvcmljYWwobW92ZV9rZXksIGxvZ2l0c1s6LCA6MTc2NV0pLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBqYXgucmFuZG9tLmNhdGVnb3JpY2FsKHNwbGl0X2tleSwgbG9naXRzWzosIDE3NjU6XSkpLCBheGlzPS0xKQogICAgICAgIHJldHVybiBhY3Rpb25zLCBzdGF0ZQo=')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
out=$disk/native-hint-prior-parity-$SLURM_JOB_ID
mkdir -p "$out"
build=$disk/native-hint-prior-build-26725/build
python3 - "$build/build.json" "$out/training.json" <<'PYINNER'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps(dict(build=manifest, config=dict(overrides={"policy.hidden_size":512,"policy.num_layers":0})))+"\n")
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
name=relh-classic-native-hint-prior-parity-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery" -v "$out:/recovery-output" \
    -v "$disk/native_puffer_policy-fixed-prior-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v "$disk/native_hint_initializer-feedforward-20260927.py:/work/source-normalized-16245/integrations/native_hint_initializer.py:ro" \
    -v "$disk/metta_puffer-terminal-boundary-fixed-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b bash -c '
      python -m integrations.native_hint_initializer --build /recovery/native-hint-prior-build-26725/build/build.json --output /recovery-output/initializer --layers 0 &&
      python /recovery/verify_native_puffer_policy-feedforward-20260927.py \
        --source /recovery/native-hint-prior-build-26725/build/source \
        --build /recovery/native-hint-prior-build-26725/build/build.json \
        --training /recovery-output/training.json \
        --checkpoint /recovery-output/initializer/policy.bin \
        --sha256 $(sha256sum /recovery-output/initializer/policy.bin | cut -d " " -f1) \
        --output /recovery-output/proof --public-views' > "$out/container.log" 2>&1
cat "$out/proof/parity.json"
NODE

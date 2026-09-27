#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-terminal-boundary-proof
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-terminal-boundary-proof-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" a7291d2a2f25166941b07309789865822e85ccbef721edbb199bc8fb3520e3ba \
    "$disk/metta_puffer-reward-scaled-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-terminal-boundary-proof-$job
mkdir -p "$out"
python3 - "$disk" <<"PY"
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/"verify_device_terminal_boundary-20260927.py"
data=base64.b64decode("IiIiUmVwcm9kdWNlIGJhdGNoLXdpbmRvdyB0ZXJtaW5hbCBmbGFncyBvbiB2YWxpZCBmcmVzaCBnYW1lIHN0YXRlcyBvbiBHUFUuIiIiCgppbXBvcnQgYXJncGFyc2UKaW1wb3J0IGpzb24KZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCgppbXBvcnQgamF4CmltcG9ydCBqYXgubnVtcHkgYXMgam5wCmltcG9ydCBudW1weSBhcyBucApmcm9tIG1ldHRhX3RyYWluaW5nLmVudmlyb25tZW50IGltcG9ydCBFbnZpcm9ubWVudENvbnRleHQKCmZyb20gaW50ZWdyYXRpb25zLm1ldHRhX3B1ZmZlciBpbXBvcnQgQmF0Y2hlZEdlbmVyYWxzUHVmZmVyRW52aXJvbm1lbnQKCnBhcnNlciA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKCkKcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1idWlsZCIsIHR5cGU9UGF0aCwgcmVxdWlyZWQ9VHJ1ZSkKcGFyc2VyLmFkZF9hcmd1bWVudCgiLS1vdXRwdXQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCmFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCmFzc2VydCBqYXguZGV2aWNlcygpWzBdLnBsYXRmb3JtID09ICJncHUiCmFyZ3Mub3V0cHV0Lm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9RmFsc2UpCm9wdGlvbnMgPSBqc29uLmxvYWRzKGFyZ3MuYnVpbGQucmVhZF90ZXh0KCkpWyJjb25maWciXVsicHl0aG9uX2Vudmlyb25tZW50Il1bIm9wdGlvbnMiXQpvcHRpb25zLnVwZGF0ZShwYXJhbGxlbF9nYW1lcz02NCkKZW52ID0gQmF0Y2hlZEdlbmVyYWxzUHVmZmVyRW52aXJvbm1lbnQoCiAgICBjb250ZXh0PUVudmlyb25tZW50Q29udGV4dChzZWVkPTEzODAsIGluZGV4PTAsIG1vZGU9InRyYWluIiwgb3V0cHV0PWFyZ3Mub3V0cHV0KSwgKipvcHRpb25zCikKdHJ5OgogICAgdmFsdWVzLCBtYXNrcyA9IGVudi5yZXNldF9kZXZpY2UoIjEzODA6Ym91bmRhcnkiKQogICAgcGxhbmVzID0gbnAuYXNhcnJheSh2YWx1ZXMpLnJlc2hhcGUoNjQsIDE0LCA0NDEpCiAgICBtb3ZlcyA9IHBsYW5lc1s6LCA0OjhdLnJlc2hhcGUoNjQsIDE3NjQpLmFyZ21heChheGlzPTEpCiAgICBtb3ZlcyA9IG5wLndoZXJlKHBsYW5lc1s6LCAzLCAwXSA+IDAsIDE3NjQsIG1vdmVzKQogICAgYWN0aW9ucyA9IGpucC5hc2FycmF5KG5wLnN0YWNrKChtb3ZlcywgcGxhbmVzWzosIDIsIDBdID4gMCksIGF4aXM9MSksIGR0eXBlPWpucC5pbnQzMikKICAgIGFzc2VydCBucC5hc2FycmF5KG1hc2tzKVtucC5hcmFuZ2UoNjQpLCBtb3Zlc10uYWxsKCkKICAgICMgRnJlc2ggc3RhdGVzIGFyZSB2YWxpZCB3aGVuIGFuIGluZGl2aWR1YWwgZ2FtZSBoYXMganVzdCByZWN5Y2xlZC4KICAgIGVudi50dXJuID0gZW52Lmhvcml6b24gLSAxCiAgICBvcmFjbGUgPSBlbnYuX2FkdmFuY2VfZGV2aWNlX3N0YXRlcygKICAgICAgICBlbnYuc3RhdGVzLCBlbnYuYmFzZS5wb29sLCBlbnYuc2lkZXMsIGVudi5vcHBvbmVudF9pZHMsCiAgICAgICAgYWN0aW9uc1s6LCAwXSwgYWN0aW9uc1s6LCAxXSwgZW52LmtleXMsCiAgICApCiAgICBfLCBfLCBfLCByZXR1cm5lZF90ZXJtaW5hbHMsIGVwaXNvZGVfZG9uZSA9IGVudi5zdGVwX2RldmljZShhY3Rpb25zKQogICAgYWN0dWFsID0gbnAuYXNhcnJheShvcmFjbGVbLTFdLCBib29sKQogICAgcmV0dXJuZWQgPSBucC5hc2FycmF5KHJldHVybmVkX3Rlcm1pbmFscywgYm9vbCkKICAgIGFzc2VydCBub3QgYWN0dWFsLmFueSgpLCAiVGhlIGZyZXNoIGdhbWVzIG11c3Qgc3RpbGwgYmUgdW5maW5pc2hlZCIKICAgIGFzc2VydCBhbGwobnAuYXJyYXlfZXF1YWwobnAuYXNhcnJheShhKSwgbnAuYXNhcnJheShiKSkKICAgICAgICAgICAgICAgZm9yIGEsIGIgaW4gemlwKGpheC50cmVlLmxlYXZlcyhvcmFjbGVbMF0pLCBqYXgudHJlZS5sZWF2ZXMoZW52LnN0YXRlcyksIHN0cmljdD1UcnVlKSkKICAgIHJlc3VsdCA9IGRpY3Qoc2NvcGU9IkdQVSBiYXRjaC13aW5kb3cgbGlmZWN5Y2xlIHJlcHJvZHVjdGlvbjsgbm8gc2tpbGwgb3IgU1BTIGNsYWltIiwKICAgICAgICAgICAgICAgICAgZ2FtZXM9NjQsIGFjdHVhbF9nYW1lX3Rlcm1pbmFscz1pbnQoYWN0dWFsLnN1bSgpKSwKICAgICAgICAgICAgICAgICAgcmV0dXJuZWRfdGVybWluYWxzPWludChyZXR1cm5lZC5zdW0oKSksCiAgICAgICAgICAgICAgICAgIGFydGlmaWNpYWxfdGVybWluYWxzPWludCgocmV0dXJuZWQgJiB+YWN0dWFsKS5zdW0oKSksCiAgICAgICAgICAgICAgICAgIGVwaXNvZGVfZG9uZT1ib29sKGVwaXNvZGVfZG9uZSkpCiAgICBhc3NlcnQgcmVzdWx0WyJhcnRpZmljaWFsX3Rlcm1pbmFscyJdID09IDY0IGFuZCBlcGlzb2RlX2RvbmUKICAgIChhcmdzLm91dHB1dCAvICJwcm9vZi5qc29uIikud3JpdGVfdGV4dChqc29uLmR1bXBzKHJlc3VsdCwgaW5kZW50PTIpICsgIlxuIikKICAgIHByaW50KGpzb24uZHVtcHMocmVzdWx0KSwgZmx1c2g9VHJ1ZSkKZmluYWxseToKICAgIGVudi5jbG9zZSgpCg==")
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open("xb") as f:
        f.write(data)
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
name=relh-classic-native-policy-terminal-boundary-proof-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-reward-scaled-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/native_puffer_policy-v2-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/verify_device_terminal_boundary-20260927.py \
      --build /recovery/native-policy-reward-scaled-build-26260/build/build.json \
      --output /recovery-output/proof > "$out/container.log" 2>&1
cat "$out/proof/proof.json"

'

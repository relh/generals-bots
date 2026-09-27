#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-spatial-device-profile
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/spatial-device-profile-%j.log
set -euo pipefail
python3 - /var/tmp/relh-generals-recovery <<'STAGE'
import base64,sys
from pathlib import Path
path=Path(sys.argv[1])/'profile_coworld_spatial_device-v1-20260927.py'
data=base64.b64decode('IiIiTWVhc3VyZSBmdWxsIENsYXNzaWMgZGV2aWNlIHN0ZXBzIHdpdGggdGhlIHNwYXRpYWwgcGlsb3QncyBleGFjdCBlbnZpcm9ubWVudC4iIiIKCmltcG9ydCBhcmdwYXJzZQppbXBvcnQganNvbgppbXBvcnQgdGltZQpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBqYXgKaW1wb3J0IGpheC5udW1weSBhcyBqbnAKaW1wb3J0IG51bXB5IGFzIG5wCmZyb20gbWV0dGFfdHJhaW5pbmcuZW52aXJvbm1lbnQgaW1wb3J0IEVudmlyb25tZW50Q29udGV4dAoKZnJvbSBpbnRlZ3JhdGlvbnMubWV0dGFfcHVmZmVyIGltcG9ydCBCYXRjaGVkR2VuZXJhbHNQdWZmZXJFbnZpcm9ubWVudAoKCmRlZiBtYWluKCkgLT4gTm9uZToKICAgIHBhcnNlciA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKCkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tYnVpbGQiLCB0eXBlPVBhdGgsIHJlcXVpcmVkPVRydWUpCiAgICBwYXJzZXIuYWRkX2FyZ3VtZW50KCItLWdhbWVzIiwgdHlwZT1pbnQsIGRlZmF1bHQ9ODE5MikKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0td2FybXVwIiwgdHlwZT1pbnQsIGRlZmF1bHQ9OCkKICAgIHBhcnNlci5hZGRfYXJndW1lbnQoIi0tc3RlcHMiLCB0eXBlPWludCwgZGVmYXVsdD00OCkKICAgIGFyZ3MgPSBwYXJzZXIucGFyc2VfYXJncygpCiAgICBhc3NlcnQgamF4LmRldmljZXMoKVswXS5wbGF0Zm9ybSA9PSAiZ3B1IgogICAgb3B0aW9ucyA9IGRpY3QoanNvbi5sb2FkcyhhcmdzLmJ1aWxkLnJlYWRfdGV4dCgpKVsiY29uZmlnIl1bInB5dGhvbl9lbnZpcm9ubWVudCJdWyJvcHRpb25zIl0pCiAgICBhc3NlcnQgb3B0aW9uc1siY293b3JsZF9jbGFzc2ljIl0gYW5kIG9wdGlvbnNbImxhbmRfZ2Fpbl9yZXdhcmRfd2VpZ2h0Il0gPT0gMC4yCiAgICBhc3NlcnQgb3B0aW9uc1sidGVhY2hlciJdIGlzIE5vbmUgYW5kIG9wdGlvbnNbInNoYXBpbmdfZ2FtbWEiXSA9PSAwLjk5OQogICAgb3B0aW9uc1sicGFyYWxsZWxfZ2FtZXMiXSA9IGFyZ3MuZ2FtZXMKICAgIGVudiA9IEJhdGNoZWRHZW5lcmFsc1B1ZmZlckVudmlyb25tZW50KAogICAgICAgIGNvbnRleHQ9RW52aXJvbm1lbnRDb250ZXh0KHNlZWQ9NjkxLCBpbmRleD0wLCBtb2RlPSJ0cmFpbiIsIG91dHB1dD1QYXRoKCIvcmVjb3ZlcnkiKSksCiAgICAgICAgKipvcHRpb25zLAogICAgKQogICAgdHJ5OgogICAgICAgIF8sIG1hc2tzID0gZW52LnJlc2V0X2RldmljZSgic3BhdGlhbC1kZXZpY2UtcHJvZmlsZS02OTEiKQogICAgICAgIGR1cmF0aW9ucyA9IFtdCiAgICAgICAgZm9yIHN0ZXAgaW4gcmFuZ2UoYXJncy53YXJtdXAgKyBhcmdzLnN0ZXBzKToKICAgICAgICAgICAgc3RhcnQgPSB0aW1lLnBlcmZfY291bnRlcigpCiAgICAgICAgICAgIGluZGljZXMgPSBqbnAuYXJnbWF4KG1hc2tzWzosIDogZW52LnNwZWMuYWN0aW9uX3NpemVzWzBdXSwgYXhpcz0xKQogICAgICAgICAgICBhY3Rpb25zID0gam5wLnN0YWNrKChpbmRpY2VzLCBqbnAuemVyb3NfbGlrZShpbmRpY2VzKSksIGF4aXM9MSkKICAgICAgICAgICAgdmFsdWVzLCBtYXNrcywgcmV3YXJkcywgZG9uZSwgZXBpc29kZV9kb25lID0gZW52LnN0ZXBfZGV2aWNlKGFjdGlvbnMpCiAgICAgICAgICAgIGpheC5ibG9ja191bnRpbF9yZWFkeSgodmFsdWVzLCBtYXNrcywgcmV3YXJkcywgZG9uZSkpCiAgICAgICAgICAgIGR1cmF0aW9uID0gdGltZS5wZXJmX2NvdW50ZXIoKSAtIHN0YXJ0CiAgICAgICAgICAgIGlmIHN0ZXAgPj0gYXJncy53YXJtdXA6CiAgICAgICAgICAgICAgICBkdXJhdGlvbnMuYXBwZW5kKGR1cmF0aW9uKQogICAgICAgICAgICAgICAgcHJpbnQoanNvbi5kdW1wcyh7InN0ZXAiOiBzdGVwICsgMSwgIm1zIjogcm91bmQoZHVyYXRpb24gKiAxMDAwLCAzKSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICJwb29sX2dlbmVyYXRpb24iOiBlbnYuX3Bvb2xfZ2VuZXJhdGlvbn0pLCBmbHVzaD1UcnVlKQogICAgICAgIHRvdGFsID0gc3VtKGR1cmF0aW9ucykKICAgICAgICBwcmludChqc29uLmR1bXBzKHsiZ2FtZXMiOiBhcmdzLmdhbWVzLCAid2FybXVwIjogYXJncy53YXJtdXAsICJzdGVwcyI6IGFyZ3Muc3RlcHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgInRvdGFsX3NlY29uZHMiOiB0b3RhbCwgImVuZF90b19lbmRfZW52X3NwcyI6IGFyZ3MuZ2FtZXMgKiBhcmdzLnN0ZXBzIC8gdG90YWwsCiAgICAgICAgICAgICAgICAgICAgICAgICAgIm1lZGlhbl9tcyI6IGZsb2F0KG5wLm1lZGlhbihkdXJhdGlvbnMpKSAqIDEwMDAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgInA5MF9tcyI6IGZsb2F0KG5wLnBlcmNlbnRpbGUoZHVyYXRpb25zLCA5MCkpICogMTAwMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAibWF4X21zIjogbWF4KGR1cmF0aW9ucykgKiAxMDAwfSksIGZsdXNoPVRydWUpCiAgICBmaW5hbGx5OgogICAgICAgIGVudi5jbG9zZSgpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo=')
if path.exists():
    assert path.read_bytes()==data
else:
    with path.open('xb') as handle: handle.write(data)
STAGE
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
printf '%s  %s\n' 4cec3ba19af3b78d044715603d8318dcb1298315ffa553cbbe42dfb8f55befe1 "$disk/profile_coworld_spatial_device-v1-20260927.py" | sha256sum -c -
printf '%s  %s\n' d254db7698d29c9947659925a15d68ad72fa0aad57bf89672348f0e51980ac38 "$disk/metta_puffer-land-gain-v1-20260927.py" | sha256sum -c -
printf '%s  %s\n' 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
for i in "${!allocated_gpus[@]}"; do
  IFS=, read -r memory utilization <<< "${usage[$i]}"
  memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; break; fi
done
test -n "$gpu"
name=relh-classic-spatial-device-profile-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
timeout -k 30s 10m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
  --ulimit core=0:0 --user "$(id -u):$(id -g)" \
  -v "$w:/work" -v "$disk:/recovery" \
  -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
  -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
  -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
  -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
  -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain \
  -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain \
  -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
  -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
  relh-generals-b300:20260923b python /recovery/profile_coworld_spatial_device-v1-20260927.py \
  --build /recovery/spatial-device-bridge-build-27218/build.json --games 8192 --warmup 8 --steps 48
NODE

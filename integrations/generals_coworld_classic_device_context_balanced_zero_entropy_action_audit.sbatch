#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-balanced-zero-entropy-action-audit
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
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-balanced-zero-entropy-action-audit-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e75f1862f88847c8893344c8ff26ca0bb35ff284d085ff95b1b000f1c7833d75 \
    "$disk/metta_puffer-balanced-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-balanced-zero-entropy-action-audit-$job
mkdir -p "$out"
printf "%s  %s\n" c146e740db8b8191de11ee468a849e4603e655b52f487ee44e003ef15cedef29 \
    "$disk/audit_context_action_distributions.py" | sha256sum -c -
python3 - "$out/config.json" <<"PY"
import hashlib, json, sys
from pathlib import Path
build="/recovery/device-context-calibrated-balanced-aligned-discount-16384-build-25200/build/build.json"
policies={}
for name, directory, step in [
    ("parent", "device-context-calibrated-134m-24646", 134217728),
    ("entropy", "device-context-calibrated-balanced-aligned-discount-16384-25211", 16777216),
    ("zero_entropy", "device-context-balanced-zero-entropy-16384-25374", 16777216),
]:
    native=Path("/var/tmp/relh-generals-recovery")/directory/"run"/"checkpoints/metta_generals/run"/f"{step:016d}.bin"
    policies[name]=dict(build=build, checkpoint=f"/recovery/{directory}/run/checkpoints/metta_generals/run/{step:016d}.bin", sha256=hashlib.sha256(native.read_bytes()).hexdigest())
    if name=="parent":
        policies[name]["build"]="/recovery/device-context-calibrated-16384-build-24499/build/build.json"
config=dict(seed=1333, games=32, turns=768, parent="parent", driver="zero_entropy", environment_build=build, policies=policies, hint_scales={name:1.0 for name in policies})
Path(sys.argv[1]).write_text(json.dumps(config)+"\n")
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
name=relh-classic-device-context-balanced-zero-entropy-action-audit-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/audit_context_action_distributions.py \
    --config /recovery-output/config.json --output /recovery-output/audit > "$out/container.log" 2>&1
cat "$out/audit/audit.json"
'

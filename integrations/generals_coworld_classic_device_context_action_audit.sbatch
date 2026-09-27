#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-action-audit
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
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-action-audit-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" ea5106c8467f10465d39e6a2779cca58fee464f791e1f7f36665041a76357151 \
    "$w/metta_puffer-device-fused.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-action-audit-$job
mkdir -p "$out"
printf "%s  %s\n" c146e740db8b8191de11ee468a849e4603e655b52f487ee44e003ef15cedef29 \
    "$disk/audit_context_action_distributions.py" | sha256sum -c -
python3 - "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
parent = dict(build="/work/device-context-build-18326/build.json",
    checkpoint="/recovery/device-context-overnight-22922/run/checkpoints/metta_generals/run/0000000335544320.bin",
    sha256="081c142c3ff8b9dceb0462582b1d1e3f9b13c743c73a82e99ce798b6e26e00ca")
small = dict(build="/recovery/device-context-teacher05-build-23518/build/build.json",
    checkpoint="/recovery/device-context-teacher05-23613/run/checkpoints/metta_generals/run/0000000008388608.bin",
    sha256="1f0ee38cdee14d23cc2a54e0cfd92e27b417016b996f30aae3512aa78b2a3d4f")
wide = dict(build="/recovery/device-context-teacher05-16384-build-23743/build/build.json",
    checkpoint="/recovery/device-context-teacher05-16384-23778/run/checkpoints/metta_generals/run/0000000016777216.bin",
    sha256="859e041b1e8b44d515ff3a45fbb007bd16068d944d781564f789d0a4051a1c42")
config = dict(seed=1321, games=32, turns=768, parent="parent335m", driver="teacher05_wide",
    environment_build=wide["build"],
    policies=dict(parent335m=parent, teacher05_small=small, teacher05_wide=wide, quarter_hint=wide),
    hint_scales=dict(parent335m=1.0, teacher05_small=1.0, teacher05_wide=1.0, quarter_hint=.25))
Path(sys.argv[1]).write_text(json.dumps(config) + "\n")
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
name=relh-classic-device-context-action-audit-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-fused.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
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

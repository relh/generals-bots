#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-raw-action-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-raw-action-audit-%j.log
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

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/native-raw-action-audit-$SLURM_JOB_ID
mkdir -p "$out"
printf '%s  %s\n' 7aaa2a8b40e5bf70cd4c1d69896cf6939d0f6f4f5d591189ef7992dd022294ae \
    "$disk/audit_native_action_distributions-strata-20260927.py" | sha256sum -c -
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
prior_build = root / 'native-hint-prior-build-v2-26757/build/build.json'
entries = {
    'raw_initial': ('native-raw-ppo-134m-26780/run',
                    'checkpoints/metta_generals/run/0000000016777216.bin', raw_build),
    'raw_constant': ('native-raw-ppo-r4-noanneal-33m-26836/run',
                     'checkpoints/metta_generals/run/0000000033554432.bin', raw_build),
    'fixed_prior': ('native-fixed-prior-noanneal-33m-26820/run',
                    'initial-policy.bin', prior_build),
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
name=relh-classic-native-raw-action-audit-$SLURM_JOB_ID
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
    relh-generals-b300:20260923b python /recovery/audit_native_action_distributions-strata-20260927.py \
    --config /recovery-output/config.json --output /recovery-output/proof > "$out/container.log" 2>&1
cat "$out/proof/audit.json"
NODE

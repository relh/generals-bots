#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-calibrated-balanced134m-action-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-calibrated-balanced134m-action-audit-%j.log
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


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
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
out=$disk/device-context-calibrated-balanced134m-action-audit-$job
mkdir -p "$out"
printf "%s  %s\n" c146e740db8b8191de11ee468a849e4603e655b52f487ee44e003ef15cedef29 \
    "$disk/audit_context_action_distributions.py" | sha256sum -c -
python3 - "$out/config.json" "25200" "25236" <<"PY"
import hashlib, json, sys
from pathlib import Path
parent = dict(build="/recovery/device-context-calibrated-16384-build-24499/build/build.json",
    checkpoint="/recovery/device-context-calibrated-134m-24646/run/checkpoints/metta_generals/run/0000000134217728.bin",
    sha256="a672058d8b3abfc6d8e8430e88b5431fab786cae2bd909e55c1eb36ce901d1be")
run = Path("/var/tmp/relh-generals-recovery") / f"device-context-calibrated-balanced-aligned-discount-134m-{sys.argv[3]}" / "run"
record = json.loads((run / "completed.json").read_text())
assert Path(record["final_checkpoint"]).name == "0000000134217728.bin"
checkpoint = run / "checkpoints/metta_generals/run/0000000134217728.bin"
trained = dict(build=f"/recovery/device-context-calibrated-balanced-aligned-discount-16384-build-{sys.argv[2]}/build/build.json",
    checkpoint=f"/recovery/device-context-calibrated-balanced-aligned-discount-134m-{sys.argv[3]}/run/checkpoints/metta_generals/run/0000000134217728.bin",
    sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
config = dict(seed=1332, games=32, turns=768, parent="parent_calibrated", driver="pure_ppo",
    environment_build=trained["build"],
    policies=dict(parent_calibrated=parent, pure_ppo=trained),
    hint_scales=dict(parent_calibrated=1.0, pure_ppo=1.0))
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
name=relh-classic-device-context-calibrated-balanced134m-action-audit-$job
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

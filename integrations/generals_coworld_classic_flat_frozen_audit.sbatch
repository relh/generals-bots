#!/usr/bin/env bash
# Replace __SOURCE_BUNDLE__ with a base64 tar of the two source files.
#SBATCH --job-name=relh-classic-flat-frozen-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-flat-frozen-audit-%j.log
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

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
old=$disk/classic-flat-source-route-selfplay-pilot-29162
out=$disk/classic-flat-frozen-audit-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
source=$out/staged
build=$old/build/build.json
checkpoint=$old/run/checkpoints/metta_generals/run/0000000016777216.bin
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = 5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71
gpu=$(bash "$old/staged/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=32G"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo 'Allocated physical GPU occupied' >&2; exit 1; }
name=relh-classic-flat-frozen-audit-$SLURM_JOB_ID
trap 'docker stop -t 5 "$name" >/dev/null 2>&1 || true' EXIT
common=(--rm --gpus "device=$gpu" --cpus=4 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$old/staged/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$old/staged/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
timeout -k 20s 6m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-flat-frozen-audit-$SLURM_JOB_ID/staged/audit_coworld_flat_frozen_opponent.py" \
    --build /recovery/classic-flat-source-route-selfplay-pilot-29162/build/build.json \
    --checkpoint /recovery/classic-flat-source-route-selfplay-pilot-29162/run/checkpoints/metta_generals/run/0000000016777216.bin \
    --sha256 5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71 \
    --output "/recovery/classic-flat-frozen-audit-$SLURM_JOB_ID/result" > "$out/audit.log" 2>&1
cat "$out/result/audit.json"
NODE

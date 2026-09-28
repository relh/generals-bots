#!/usr/bin/env bash
# Replace bundle marker with base64 tar of the exact evaluator and adapter.
#SBATCH --job-name=relh-classic-nohint-teacher-action-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-nohint-teacher-action-audit-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
w=/tmp/relh-generals-coworld
out=$disk/classic-nohint-teacher-action-audit-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
sha256sum "$out/staged"/*
gpu=$(bash "$out/staged/allocated_gpu_uuid.sh")
echo "node=$(hostname) job=$SLURM_JOB_ID gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU occupied" >&2; exit 1; }
echo "docker_root=$(docker info --format '{{.DockerRootDir}}')"
df -h /var/lib/docker "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
source=$disk/classic-nohint-global-small-pilot-28483
test -f "$source/run/completed.json"
checkpoint=$source/run/checkpoints/metta_generals/run/0000000033554432.bin
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = f075b076cb428b69049e8ce3975284f5b69f837e27c94ec7c59541d26aaf0609
name=relh-classic-nohint-teacher-action-audit-$SLURM_JOB_ID
trap 'docker stop -t 5 "$name" >/dev/null 2>&1 || true' EXIT
common=(--rm --gpus "device=$gpu" --cpus=4 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v "$out/staged/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$out/staged/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
container_uuid=$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)
test "$container_uuid" = "$gpu"
timeout -k 30s 6m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-nohint-teacher-action-audit-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_opponent_pilot.py" \
    --build /recovery/classic-nohint-global-small-pilot-28483/build/build.json \
    --run /recovery/classic-nohint-global-small-pilot-28483/run \
    --checkpoint /recovery/classic-nohint-global-small-pilot-28483/run/checkpoints/metta_generals/run/0000000033554432.bin \
    --sha256 f075b076cb428b69049e8ce3975284f5b69f837e27c94ec7c59541d26aaf0609 \
    --games 128 --pool-size 128 --seed 1386 --opponent expander_harvester \
    --teacher-action-audit --audit-turns 128 \
    --output "/recovery/classic-nohint-teacher-action-audit-$SLURM_JOB_ID/score" > "$out/audit.log" 2>&1
python3 - "$out/score/evaluation.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(json.dumps(r['teacher_action_audit'],sort_keys=True),flush=True)
PY
NODE

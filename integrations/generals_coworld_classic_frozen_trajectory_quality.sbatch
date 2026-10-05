#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-frozen-trajectory-quality
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-frozen-trajectory-%j.log
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

srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
pilot=$disk/classic-frozen-pilot-28247
source=$pilot/staged
out=$disk/classic-frozen-trajectory-$SLURM_JOB_ID
mkdir -p "$out"
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
test -f "$pilot/run/completed.json"
test -f "$source/evaluate_coworld_frozen_opponent_pilot.py"
test -f "$pilot/build/build.json"
name=relh-classic-frozen-trajectory-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro"
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
for step in 2097152 6291456 10485760; do
    checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$pilot" "$step")
    digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
    for opponent in expander_harvester sentinel; do
        tag=$step-$opponent
        timeout -k 30s 4m docker run "${common[@]}" --name "$name" \
            relh-generals-b300:20260923b \
            python "/recovery/classic-frozen-pilot-28247/staged/evaluate_coworld_frozen_opponent_pilot.py" \
            --build /recovery/classic-frozen-pilot-28247/build/build.json \
            --run /recovery/classic-frozen-pilot-28247/run \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
            --output "/recovery/classic-frozen-trajectory-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
        python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
d=json.load(open(sys.argv[1]))
print(d['checkpoint_sha256'],d['opponent'],d['wins'],d['losses'],d['draws'],d['score'],flush=True)
PYSCORE
    done
done
checkpoint=$pilot/run/checkpoints/metta_generals/run/0000000012582912.bin
digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
timeout -k 30s 4m docker run "${common[@]}" --name "$name" \
    relh-generals-b300:20260923b \
    python /recovery/classic-frozen-pilot-28247/staged/evaluate_coworld_frozen_opponent_pilot.py \
    --build /recovery/classic-frozen-pilot-28247/build/build.json \
    --run /recovery/classic-frozen-pilot-28247/run \
    --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
    --games 64 --pool-size 64 --seed 1387 --opponent expander_harvester \
    --force-hint-move --force-hint-split --hint-audit --audit-turns 128 \
    --output "/recovery/classic-frozen-trajectory-$SLURM_JOB_ID/hint-audit" \
    > "$out/hint-audit.log" 2>&1
python3 - "$out/hint-audit/evaluation.json" <<'PYHINT'
import json,sys
d=json.load(open(sys.argv[1]))
print('hint_audit',d['hint_audit'],flush=True)
PYHINT
NODE

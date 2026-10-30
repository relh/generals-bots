#!/usr/bin/env bash
# Replace __EVALUATOR_BASE64__ with base64 of the evaluator source.
#SBATCH --job-name=relh-classic-flat-tactical-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:18:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-flat-tactical-audit-%j.log
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
disk=/var/tmp/relh-generals-recovery
flat=$disk/classic-flat-source-route-selfplay-pilot-29162
frozen=$disk/classic-flat-frozen-pilot-29209
out=$disk/classic-flat-tactical-audit-$SLURM_JOB_ID
mkdir -p "$out"
base64 -d <<'EVALUATOR' > "$out/evaluate.py"
__EVALUATOR_BASE64__
EVALUATOR
sha256sum "$out/evaluate.py"
test -f "$flat/run/completed.json" && test -f "$frozen/run/completed.json"
test "$(sha256sum "$flat/run/checkpoints/metta_generals/run/0000000016777216.bin" | cut -d' ' -f1)" = 5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71
test "$(sha256sum "$flat/run/checkpoints/metta_generals/run/0000000067108864.bin" | cut -d' ' -f1)" = 1cc05e71b8612d31c5497f5b07ec3210094e095982b349ff8ab4af0f4c7b3df5
test "$(sha256sum "$frozen/run/checkpoints/metta_generals/run/0000000012582912.bin" | cut -d' ' -f1)" = 3e0fefc7a50a48dcb9558aa8664e826aedaa24aa29330a08e5416c8753bac6e1
gpu=$(bash "$flat/staged/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo 'Allocated physical GPU occupied' >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
name=relh-classic-flat-tactical-audit-$SLURM_JOB_ID
trap 'docker stop -t 5 "$name" >/dev/null 2>&1 || true' EXIT
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$frozen/staged/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$flat/staged/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$flat/staged/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
for policy in flat_early flat_final frozen_final; do
    case $policy in
        flat_early) model=$flat; step=16777216; digest=5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71 ;;
        flat_final) model=$flat; step=67108864; digest=1cc05e71b8612d31c5497f5b07ec3210094e095982b349ff8ab4af0f4c7b3df5 ;;
        frozen_final) model=$frozen; step=12582912; digest=3e0fefc7a50a48dcb9558aa8664e826aedaa24aa29330a08e5416c8753bac6e1 ;;
    esac
    checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$model" "$step")
    for opponent in expander_harvester sentinel; do
        tag=$policy-$opponent
        timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-flat-tactical-audit-$SLURM_JOB_ID/evaluate.py" \
            --build "${model/$disk/\/recovery}/build/build.json" \
            --run "${model/$disk/\/recovery}/run" \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-flat-tactical-audit-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
        python3 - "$out/$tag/evaluation.json" <<'PYRESULT'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['action_stats'],flush=True)
PYRESULT
    done
done
NODE

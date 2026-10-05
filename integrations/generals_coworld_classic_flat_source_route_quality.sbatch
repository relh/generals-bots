#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-flat-source-route-quality
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
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-flat-source-route-quality-%j.log
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
source=$disk/classic-flat-source-route-selfplay-pilot-29162/staged
build=$disk/classic-flat-source-route-selfplay-pilot-29162/build/build.json
out=$disk/classic-flat-source-route-quality-$SLURM_JOB_ID
mkdir -p "$out"
base64 -d <<'EVALUATOR' > "$out/evaluate.py"
__EVALUATOR_BASE64__
EVALUATOR
test "$(sha256sum "$out/evaluate.py" | cut -d' ' -f1)" = 62d6919555670832955b6f6a625ef592688a517c1493333cb87aa4ce8c29389b
test "$(sha256sum "$source/generals_fabric.py" | cut -d' ' -f1)" = 445724d7330622ca44a9f81ffb4531013add2596c8eb95ce6d93141fd196c322
test "$(sha256sum "$source/puffer_codec.py" | cut -d' ' -f1)" = 42cd4893a9eafc2fb61314bccc4aa95e9940d4768aa4ab49d6fb60affd20ff52
python3 - "$build" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['model_sha256']=='c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d'
assert m['config']['fabric']['options']['source_army_prior_strength']==4.0
assert m['config']['python_environment']['spec']['action_sizes']==[8*21*21+1]
PY
checkpoints=(
    "$disk/classic-flat-source-route-selfplay-pilot-29162/run/checkpoints/metta_generals/run/0000000016777216.bin"
    "$disk/classic-flat-source-route-selfplay-pilot-29162/run/checkpoints/metta_generals/run/0000000067108864.bin"
)
expected=(
    5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71
    1cc05e71b8612d31c5497f5b07ec3210094e095982b349ff8ab4af0f4c7b3df5
)
for i in 0 1; do printf '%s  %s\n' "${expected[i]}" "${checkpoints[i]}" | sha256sum -c -; done
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G source_revision=6b2b5d3b75782bf032eacef304e43ba66dbfb974"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied; stopping" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
name=relh-classic-flat-source-route-quality-$SLURM_JOB_ID
trap 'docker stop -t 5 "$name" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$source/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$source/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
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
for i in 0 1; do
    if [[ $i == 0 ]]; then
        run=/recovery/classic-flat-source-route-selfplay-pilot-29162/run
        extra=()
        label=early
    else
        run=/recovery/classic-flat-source-route-selfplay-pilot-29162/run
        extra=()
        label=final
    fi
    checkpoint=${checkpoints[i]/$disk/\/recovery}
    for opponent in expander_harvester sentinel; do
        result=$out/$label-$opponent
        timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-flat-source-route-quality-$SLURM_JOB_ID/evaluate.py" \
            --build /recovery/classic-flat-source-route-selfplay-pilot-29162/build/build.json \
            --run "$run" --checkpoint "$checkpoint" --sha256 "${expected[i]}" "${extra[@]}" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-flat-source-route-quality-$SLURM_JOB_ID/$label-$opponent" \
            > "$out/$label-$opponent.log" 2>&1
        python3 - "$result/evaluation.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['action_stats'],flush=True)
PY
    done
done
NODE

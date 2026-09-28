#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-source-route-action-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:18:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-source-route-action-audit-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
source=$disk/classic-source-route-selfplay-pilot-29068/staged
build=$disk/classic-source-route-selfplay-pilot-29068/build/build.json
out=$disk/classic-source-route-action-audit-$SLURM_JOB_ID
mkdir -p "$out"
base64 -d <<'EVALUATOR' > "$out/evaluate.py"
__EVALUATOR_BASE64__
EVALUATOR
test "$(sha256sum "$out/evaluate.py" | cut -d' ' -f1)" = 6f83968995dec613786c5e158722ea2b4c70876fbd9415566624cf5b262f0790
python3 - "$build" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
assert m['model_sha256']=='e7c3b4366f27cd4460ab3b037989bcc66596d5db69931ee4eeb2968df0702a8b'
assert m['config']['fabric']['options']['source_army_prior_strength']==4.0
PY
checkpoints=(
    "$disk/classic-source-route-selfplay-pilot-29068/run/checkpoints/metta_generals/run/0000000016777216.bin"
    "$disk/classic-source-route-resume-pilot-29117/run/checkpoints/metta_generals/run/0000000067108864.bin"
)
expected=(
    9dbba5ad998f9904ca58f0b445d9a859ee9f4748d9a405d425ac86822d990e07
    b580c5260bd29af14c25d72303e546b42ccb36a894e1408846dc95694b298943
)
for i in 0 1; do printf '%s  %s\n' "${expected[i]}" "${checkpoints[i]}" | sha256sum -c -; done
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G source_revision=06c0f01c23aa8fc70bc9ea8378fcce6ee28b5814"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied; stopping" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
name=relh-classic-source-route-action-audit-$SLURM_JOB_ID
trap 'docker stop -t 5 "$name" >/dev/null 2>&1 || true' EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$source/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
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
for i in 0 1; do
    if [[ $i == 0 ]]; then
        run=/recovery/classic-source-route-selfplay-pilot-29068/run
        extra=(--diagnostic-checkpoint)
        label=early
    else
        run=/recovery/classic-source-route-resume-pilot-29117/run
        extra=()
        label=final
    fi
    checkpoint=${checkpoints[i]/$disk/\/recovery}
    for opponent in expander_harvester sentinel; do
        result=$out/$label-$opponent
        timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-source-route-action-audit-$SLURM_JOB_ID/evaluate.py" \
            --build /recovery/classic-source-route-selfplay-pilot-29068/build/build.json \
            --run "$run" --checkpoint "$checkpoint" --sha256 "${expected[i]}" "${extra[@]}" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-source-route-action-audit-$SLURM_JOB_ID/$label-$opponent" \
            > "$out/$label-$opponent.log" 2>&1
        python3 - "$result/evaluation.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['action_stats'],flush=True)
PY
    done
done
NODE

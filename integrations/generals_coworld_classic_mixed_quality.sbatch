#!/usr/bin/env bash
# Bundle marker is replaced with base64 tar of the corrected evaluator.
#SBATCH --job-name=relh-classic-mixed-quality
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-mixed-quality-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
pilot=$disk/classic-mixed-pilot-28306
source=$pilot/staged
out=$disk/classic-mixed-quality-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__EVALUATOR_BUNDLE__
BUNDLE
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
test -f "$pilot/run/completed.json"
test -f "$pilot/build/build.json"
name=relh-classic-mixed-quality-$SLURM_JOB_ID
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
checkpoint=$pilot/run/checkpoints/metta_generals/run/0000000012582912.bin
digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
for opponent in expander_harvester sentinel; do
    timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
        python "/recovery/classic-mixed-quality-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_opponent_pilot.py" \
        --build /recovery/classic-mixed-pilot-28306/build/build.json \
        --run /recovery/classic-mixed-pilot-28306/run \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
        --output "/recovery/classic-mixed-quality-$SLURM_JOB_ID/$opponent" > "$out/$opponent.log" 2>&1
    python3 - "$out/$opponent/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
done
NODE

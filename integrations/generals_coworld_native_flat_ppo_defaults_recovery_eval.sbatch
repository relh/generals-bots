#!/usr/bin/env bash
# Replace __SOURCE_BUNDLE__ with pinned source files, including puffer_advantage_normalization.cuh.
# Diagnostic only: Stock Puffer5 MinGRU, multi-step capture and actual flat legal masks.
#SBATCH --job-name=relh-classic-native-flat-defaults-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-native-flat-ppo-defaults-eval-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
old=$disk/classic-flat-source-route-selfplay-pilot-29162
parent=$disk/classic-flat-scripted-potential-pilot-29432
out=$disk/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID
mkdir -p "$out/staged"
original=$disk/classic-native-flat-ppo-defaults-pilot-29836
cp -a "$original/staged/." "$out/staged/"
cat "$original/native-parity.log"
source=$out/staged
sha256sum "$source"/*.py "$source"/*.cuh
checkpoint=$old/run/checkpoints/metta_generals/run/0000000016777216.bin
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = 5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71
test -f "$old/run/completed.json"
gpu=$(bash "$old/staged/allocated_gpu_uuid.sh")
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
name=relh-classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID
sampler=
cleanup() {
    test -z "$sampler" || kill "$sampler" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v /tmp/relh-generals-coworld:/work -v "$disk:/recovery"
    -v "$source/metta_puffer.py:/work/source-spatial2/integrations/metta_puffer.py:ro"
    -v "$source/generals_fabric.py:/work/source-spatial2/integrations/generals_fabric.py:ro"
    -v "$source/native_puffer_policy.py:/work/source-spatial2/integrations/native_puffer_policy.py:ro"
    -v "$source/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e METTA_AUDIT_ACTION_MASK=1
    -e PYTHONPATH=/recovery/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID/staged:/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
checkpoint=$original/run/checkpoints/metta_generals/run/0000000033554432.bin
digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
timeout -k 30s 3m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID/staged/verify_native_puffer_policy.py" \
    --source "/recovery/classic-native-flat-ppo-defaults-pilot-29836/build/source" \
    --build "/recovery/classic-native-flat-ppo-defaults-pilot-29836/build/build.json" \
    --training "/recovery/classic-native-flat-ppo-defaults-pilot-29836/run/training.json" \
    --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" --public-views \
    --output "/recovery/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID/native-parity" > "$out/native-parity.log" 2>&1
for step in 8388608 33554432; do
    for opponent in expander_harvester sentinel; do
        checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$original" "$step")
        digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
        tag=$step-$opponent
        timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_greedy.py" --native \
            --build "/recovery/classic-native-flat-ppo-defaults-pilot-29836/build/build.json" \
            --run "/recovery/classic-native-flat-ppo-defaults-pilot-29836/run" \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-native-flat-ppo-defaults-eval-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
        python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
    done
done
NODE

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-source-route-resume
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-source-route-resume-pilot-%j.log
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
source=$disk/classic-source-route-selfplay-pilot-29068/staged
parent=$disk/classic-source-route-selfplay-pilot-29068
checkpoint=$parent/run/checkpoints/metta_generals/run/0000000016777216.bin
checkpoint_sha=9dbba5ad998f9904ca58f0b445d9a859ee9f4748d9a405d425ac86822d990e07
out=$disk/classic-source-route-resume-pilot-$SLURM_JOB_ID
mkdir -p "$out"
for path in "$checkpoint" "$checkpoint.learner" "$checkpoint.learner.json" "$parent/run/training.json" "$parent/config.json" "$parent/build/build.json" "$parent/build/puffer"; do test -f "$path"; done
printf '%s  %s\n' "$checkpoint_sha" "$checkpoint" | sha256sum -c -
test "$(sha256sum "$source/launch_classic_frozen_opponent_training.py" | cut -d' ' -f1)" = 623ae4927e81db858fc8450b5ef3dbafadb1327782f1dfdff7ac3f38df4f48a6
python3 - "$parent/config.json" "$out/config.json" "$parent/run/training.json" <<'PY'
import json,sys
from pathlib import Path
source,target,record=map(Path,sys.argv[1:])
run=json.loads(source.read_text())
training=json.loads(record.read_text())
assert run['seed']==training['config']['seed']==732
assert run['overrides']==training['config']['overrides']
assert run['total_timesteps']==67_108_864
assert training['build']['model_sha256']=='e7c3b4366f27cd4460ab3b037989bcc66596d5db69931ee4eeb2968df0702a8b'
assert run.get('initialize') is None
run['initialize']={
    'run':'/recovery/classic-source-route-selfplay-pilot-29068/run',
    'checkpoint':'/recovery/classic-source-route-selfplay-pilot-29068/run/checkpoints/metta_generals/run/0000000016777216.bin',
    'sha256':'9dbba5ad998f9904ca58f0b445d9a859ee9f4748d9a405d425ac86822d990e07',
    'restore_learner':True,
}
target.write_text(json.dumps(run)+'\n')
PY
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G source_revision=56f97df44e4fa6a751b1de31a676151fa6988097 checkpoint_sha=$checkpoint_sha"
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied; stopping" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
name=relh-classic-source-route-resume-$SLURM_JOB_ID
sampler=
cleanup() {
    test -z "$sampler" || kill "$sampler" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 143' TERM
common=(--rm --gpus "device=$gpu" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
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
ln -s "$out/run" "$disk/classic-source-route-resume-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-source-route-resume-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 28m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-source-route-resume-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-source-route-selfplay-pilot-29068/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-source-route-resume-pilot-$SLURM_JOB_ID/config.json" \
    --build /recovery/classic-source-route-selfplay-pilot-29068/build \
    --output "/recovery/classic-source-route-resume-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-source-route-resume 600 1048576 60000; then
        echo "Source-route resume stopped by environment throughput or progress guard" >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$disk/monitor_coworld_steady_interval-v2.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-source-route-resume 600 1048576 60000
for step in 33554432 67108864; do
for opponent in expander_harvester sentinel; do
    checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
    digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
    tag=$step-$opponent
    timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
        python "/recovery/classic-source-route-selfplay-pilot-29068/staged/evaluate_coworld_frozen_opponent_pilot.py" \
        --build /recovery/classic-source-route-selfplay-pilot-29068/build/build.json \
        --run "/recovery/classic-source-route-resume-pilot-$SLURM_JOB_ID/run" \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
        --output "/recovery/classic-source-route-resume-pilot-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
    python3 - "$out/$tag/evaluation.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PY
done
done
NODE

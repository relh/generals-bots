#!/usr/bin/env bash
# Bundle marker is replaced with base64 tar of the source files at submission.
#SBATCH --job-name=relh-classic-gen0-long-selfplay
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-gen0-long-pilot-%j.log
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
out=$disk/classic-gen0-long-pilot-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
source=$out/staged
sha256sum "$source"/*
gpu=$(bash "$source/allocated_gpu_uuid.sh")
echo "job=$SLURM_JOB_ID node=$(hostname) partition=$SLURM_JOB_PARTITION gpu_uuid=$gpu cpus=$SLURM_CPUS_PER_TASK mem=64G"
echo "source_revision=$(git -C "$w/source-spatial2" rev-parse HEAD 2>/dev/null || echo snapshot)"
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,compute_cap,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader
IFS=, read -r uuid memory utilization <<< "$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
uuid=${uuid//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
test "$uuid" = "$gpu"
(( memory < 2048 && utilization < 20 )) || { echo "Allocated physical GPU is occupied; stopping" >&2; exit 1; }
docker_root=$(docker info --format '{{.DockerRootDir}}')
echo "docker_root=$docker_root"
df -h "$docker_root" "$disk"
docker image inspect relh-generals-b300:20260923b --format '{{.Id}}'
test -f "$w/source-spatial2/integrations/run_puffer_gpu.sh"
test -f "$w/metta-device-wide-source.tar.gz"
test -f "$disk/classic-selfplay-iter1-build-28048/build/build.json"
test -f "$disk/classic-selfplay-init134-h128-27957/run/completed.json"
printf '%s  %s\n' e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf \
    "$disk/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin" | sha256sum -c -
name=relh-classic-gen0-long-pilot-$SLURM_JOB_ID
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
docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    bash -c 'nvcc --version | tail -4; python -c "import jax, metta_training, generals; print(jax.devices(\"gpu\"))"' \
    > "$out/runtime.log" 2>&1
cat "$out/runtime.log"
python3 - "$disk/classic-selfplay-teacher-h128-build-27857/build.json" \
    "$disk/classic-selfplay-iter1-build-28048/build/build.json" "$out/build-config.json" \
    "$disk/classic-selfplay-iter1-calibrated-h128-28074/config.json" "$out/config.json" <<'PYCONFIG'
import json,sys
from pathlib import Path
source,teacher_target,target,run_source,run_target=map(Path,sys.argv[1:])
manifest=json.loads(source.read_text())
config=manifest['config']
assert manifest['model_sha256']=='a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd'
env=config['python_environment']
assert env['factory']=='integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment'
assert env['spec']['agents']==8192 and env['options']['parallel_games']==4096
assert 'move_hint_scale' not in env['options'] and 'split_hint_scale' not in env['options']
target_manifest=json.loads(teacher_target.read_text())
assert target_manifest['model_sha256']=='4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67'
config['fabric']['teacher']=target_manifest['config']['fabric']['teacher']
assert config['fabric']['teacher']['phases'][0]['coefficient']==0
env['factory']='integrations.metta_puffer:BatchedGeneralsFrozenOpponentPufferEnvironment'
env['spec']['agents']=4096
env['options'].update(
    frozen_build='/recovery/classic-selfplay-teacher-h128-build-27857/build.json',
    frozen_checkpoint='/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin',
    frozen_sha256='e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf',
)
target.write_text(json.dumps(config)+'\n')
run=json.loads(run_source.read_text())
run['seed']=743
run['total_timesteps']=301_989_888
run['overrides']['vec.total_agents']=4096
run['overrides']['train.minibatch_size']=32768
run['overrides']['base.checkpoint_interval']=192
assert run['overrides']['train.horizon']==128
assert run['overrides']['train.gamma']==0.999
run['initialize']={
    'run':'/recovery/classic-selfplay-init134-h128-27957/run',
    'checkpoint':'/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin',
    'sha256':'e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf',
    'allow_environment_transfer':True,
    'allow_policy_only_transfer':True,
}
run_target.write_text(json.dumps(run)+'\n')
PYCONFIG
timeout -k 30s 8m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/build-config.json" \
    --output "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<'PYBUILD'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['model_sha256']=='4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67'
assert d['config']['python_environment']['spec']['agents']==4096
print('target_model_sha256='+d['model_sha256'],flush=True)
PYBUILD
ln -s "$out/run" "$disk/classic-gen0-long-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-gen0-long-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 95m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/config.json" \
    --build "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/build" \
    --output "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-gen0-long 600 524288 30000 8 12 20; then
        echo "Generation-0 self-play run stopped by throughput or progress guard" >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$disk/monitor_coworld_steady_interval-v2.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-gen0-long 600 524288 30000 8 12 20
for step in 100663296 201326592 301989888; do
for opponent in expander_harvester sentinel; do
    checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
    digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
    tag=$step-$opponent
    timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
        python "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_opponent_pilot.py" \
        --build "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/build/build.json" \
        --run "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/run" \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
        --output "/recovery/classic-gen0-long-pilot-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
    python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
done
done
NODE

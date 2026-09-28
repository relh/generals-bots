#!/usr/bin/env bash
# Bundle marker is replaced with base64 tar of the source files at submission.
#SBATCH --job-name=relh-classic-nohint-dagger-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-nohint-dagger-pilot-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
out=$disk/classic-nohint-dagger-pilot-$SLURM_JOB_ID
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
name=relh-classic-nohint-dagger-pilot-$SLURM_JOB_ID
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
timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/staged/audit_coworld_nohint_teacher_transport.py" \
    --nohint-build /recovery/classic-nohint-selfplay-pilot-28380/build/build.json \
    --output "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/teacher-audit" \
    > "$out/teacher-audit.log" 2>&1
tail -5 "$out/teacher-audit.log"
python3 - "$disk/classic-selfplay-iter1-build-28048/build/build.json" \
    "$disk/classic-selfplay-teacher-h128-build-27857/build.json" "$out/build-config.json" \
    "$disk/classic-selfplay-iter1-calibrated-h128-28074/config.json" "$out/config.json" <<'PYCONFIG'
import json,sys
from pathlib import Path
source,teacher_source,target,run_source,run_target=map(Path,sys.argv[1:])
manifest=json.loads(source.read_text())
config=manifest['config']
assert manifest['model_sha256']=='4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67'
env=config['python_environment']
assert env['factory']=='integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment'
assert env['spec']['agents']==8192 and env['options']['parallel_games']==4096
assert env['spec']['observation_size']==14*21*21
assert config['fabric']['teacher']['phases'][0]['coefficient']==0
teacher_manifest=json.loads(teacher_source.read_text())
teacher=teacher_manifest['config']['fabric']['teacher']
assert len(teacher['phases'])==2 and teacher['phases'][0]['action_mix']==1.0
teacher['phases'][0]['action_mix']=0.5
teacher['phases'][1]['agent_steps']=100_663_296
teacher['phases'][1]['coefficient']=0.0
teacher['phases'][1]['ppo_coefficient']=1.0
teacher['phases'][1]['action_mix']=0.0
env['spec']['observation_size']=8*21*21
env['spec']['teacher']=True
env['options'].update(
    teacher='expander_harvester', supervise_teacher=True, hint_features=False,
    prior_hint_features=False, expander_hint_features=False,
    context_hint_features=False, deduplicate_opponent_branches=False,
    coworld_pool_size=256,
)
env['options'].pop('move_hint_scale',None)
env['options'].pop('split_hint_scale',None)
config['fabric']['observation_size']=8*21*21
config['fabric']['options']['channels']=8
config['fabric']['options']['features_per_site']=4
config['fabric']['options']['global_features']=8
config['fabric']['options']['broadcast_global_context']=True
config['fabric']['options']['hint_prior_strength']=0.0
config['fabric']['teacher']=teacher
target.write_text(json.dumps(config)+'\n')
run=json.loads(run_source.read_text())
run['seed']=739
run['total_timesteps']=100_663_296
run['overrides']['vec.total_agents']=8192
run['overrides']['train.minibatch_size']=32768
run['overrides']['train.ent_coef']=0.003
run['overrides']['base.checkpoint_interval']=16
assert run['overrides']['train.horizon']==128
assert run['overrides']['train.gamma']==0.999
run['initialize']={
    'run':'/recovery/classic-nohint-global-small-pilot-28483/run',
    'checkpoint':'/recovery/classic-nohint-global-small-pilot-28483/run/checkpoints/metta_generals/run/0000000033554432.bin',
    'sha256':'f075b076cb428b69049e8ce3975284f5b69f837e27c94ec7c59541d26aaf0609',
    'allow_environment_transfer':True,
    'allow_policy_only_transfer':True,
}
run_target.write_text(json.dumps(run)+'\n')
PYCONFIG
timeout -k 30s 8m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/build-config.json" \
    --output "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<'PYBUILD'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['config']['python_environment']['spec']['agents']==8192
assert d['config']['python_environment']['spec']['observation_size']==8*21*21
assert d['config']['python_environment']['spec']['teacher'] is True
assert d['config']['fabric']['teacher']['phases'][0]['action_mix']==0.5
assert d['config']['fabric']['teacher']['phases'][1]['coefficient']==0
assert d['config']['fabric']['factory']=='integrations.generals_fabric:two_stage_tied_local_action_policy'
assert d['config']['fabric']['options']['features_per_site']==4
assert d['config']['fabric']['options']['global_features']==8
assert d['config']['fabric']['options']['broadcast_global_context'] is True
print('target_model_sha256='+d['model_sha256'],flush=True)
PYBUILD
ln -s "$out/run" "$disk/classic-nohint-dagger-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-nohint-dagger-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 30m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/config.json" \
    --build "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/build" \
    --output "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-nohint-dagger 600 1048576 60000 8 12 20; then
        echo "No-hint data-aggregation pilot stopped by environment throughput or progress guard" >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$disk/monitor_coworld_steady_interval-v2.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-nohint-dagger 600 1048576 60000 8 12 20
for step in 33554432 100663296; do
for opponent in expander_harvester sentinel; do
    checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
    digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
    tag=$step-$opponent
    timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
        python "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_opponent_pilot.py" \
        --build "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/build/build.json" \
        --run "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/run" \
        --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
        --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
        --output "/recovery/classic-nohint-dagger-pilot-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
    python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
done
done
NODE

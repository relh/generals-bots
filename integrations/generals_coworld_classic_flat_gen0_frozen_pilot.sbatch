#!/usr/bin/env bash
# Replace __SOURCE_BUNDLE__ with a base64 tar of the pinned source files.
#SBATCH --job-name=relh-classic-flat-gen0-frozen-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:35:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-flat-gen0-frozen-pilot-%j.log
set -euo pipefail
srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -s <<'NODE'
set -euo pipefail
disk=/var/tmp/relh-generals-recovery
old=$disk/classic-flat-source-route-selfplay-pilot-29162
out=$disk/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
source=$out/staged
sha256sum "$source"/*
checkpoint=$old/run/checkpoints/metta_generals/run/0000000016777216.bin
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = 5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71
gen0=$disk/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin
test "$(sha256sum "$gen0" | cut -d' ' -f1)" = e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf
test "$(sha256sum "$disk/classic-flat-gen0-frozen-audit-29311/staged/legacy/generals_fabric.py" | cut -d' ' -f1)" = 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992
test -f "$disk/classic-flat-gen0-frozen-audit-29311/result/audit.json"
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
name=relh-classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID
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
    -v "$source/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
python3 - "$old/build/build.json" "$out/build-config.json" "$old/config.json" "$out/config.json" <<'PYCONFIG'
import json,sys
from pathlib import Path
source,target,run_source,run_target=map(Path,sys.argv[1:])
manifest=json.loads(source.read_text())
config=manifest['config']
assert manifest['model_sha256']=='c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d'
env=config['python_environment']
assert env['factory']=='integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment'
assert env['spec']['agents']==8192 and env['options']['parallel_games']==4096
assert env['spec']['action_sizes']==[3529] and env['options']['factorized_actions'] is False
assert env['options']['shaping_gamma']==0.999 and env['options']['balance_opponent_sides'] is True
env['factory']='integrations.metta_puffer:BatchedGeneralsFrozenOpponentPufferEnvironment'
env['spec']['agents']=4096
env['options'].update(
    frozen_build='/recovery/classic-selfplay-teacher-h128-build-27857/build.json',
    frozen_checkpoint='/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin',
    frozen_sha256='e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf',
    frozen_codec='hinted_gen0',
    frozen_legacy_fabric='/recovery/classic-flat-gen0-frozen-audit-29311/staged/legacy/generals_fabric.py',
)
target.write_text(json.dumps(config)+'\n')
run=json.loads(run_source.read_text())
assert 'initialize' not in run
run['seed']=735
run['total_timesteps']=12_582_912
run['overrides']['vec.total_agents']=4096
run['overrides']['base.checkpoint_interval']=8
assert run['overrides']['train.horizon']==128
assert run['overrides']['train.gamma']==0.999
run['initialize']={
    'run':'/recovery/classic-flat-source-route-selfplay-pilot-29162/run',
    'checkpoint':'/recovery/classic-flat-source-route-selfplay-pilot-29162/run/checkpoints/metta_generals/run/0000000016777216.bin',
    'sha256':'5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71',
    'allow_environment_transfer':True,
}
run_target.write_text(json.dumps(run)+'\n')
PYCONFIG
timeout -k 30s 8m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/build-config.json" \
    --output "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<'PYBUILD'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['model_sha256']=='c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d'
assert d['config']['python_environment']['spec']['agents']==4096
assert d['config']['python_environment']['spec']['action_sizes']==[3529]
print('target_model_sha256='+d['model_sha256'],flush=True)
PYBUILD
ln -s "$out/run" "$disk/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-flat-gen0-frozen-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 20m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/config.json" \
    --build "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/build" \
    --output "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$source/monitor_coworld_steady_interval.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-flat-frozen 600 524288 30000 8 12 20; then
        echo 'Flat frozen pilot stopped by throughput or progress guard' >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$source/monitor_coworld_steady_interval.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-flat-frozen 600 524288 30000 8 12 20
for step in 4194304 12582912; do
    for opponent in expander_harvester sentinel; do
        checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
        digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
        tag=$step-$opponent
        timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_opponent_pilot.py" \
            --build "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/build/build.json" \
            --run "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/run" \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-flat-gen0-frozen-pilot-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
        python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
    done
done
NODE

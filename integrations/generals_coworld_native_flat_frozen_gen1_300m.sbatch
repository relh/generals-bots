#!/usr/bin/env bash
# Replace __SOURCE_BUNDLE__ with pinned source files, including puffer_advantage_normalization.cuh.
# 300M additional physical steps, same proven native frozen-generation geometry.
#SBATCH --job-name=relh-classic-native-flat-frozen-gen1-300m-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-native-flat-frozen-gen1-300m-pilot-%j.log
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
parent=$disk/classic-native-flat-frozen-gen1-pilot-29937
out=$disk/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
source=$out/staged
sha256sum "$source"/*.py "$source"/*.cuh
checkpoint=$parent/run/checkpoints/metta_generals/run/0000000033554432.bin
test "$(sha256sum "$checkpoint" | cut -d' ' -f1)" = 3ab08100c02dc06a3ffa3888e26487dc87df2d41408c43ae854ac47088d6d3af
test -f "$parent/run/completed.json"
gpu=$(bash "$source/allocated_gpu_uuid.sh")
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
name=relh-classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID
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
    -v "$source/native_selfplay.py:/work/source-spatial2/integrations/native_selfplay.py:ro"
    -v "$source/puffer_codec.py:/work/source-spatial2/integrations/puffer_codec.py:ro"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e METTA_AUDIT_ACTION_MASK=1
    -e PYTHONPATH=/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/staged:/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
ln -s "../${parent##*/}/build" "$out/build"
python3 - "$parent/run/training.json" "$out/config.json" <<'PYCONFIG'
import copy,json,sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text())
assert record['build']['config']['fabric'] is None
assert record['build']['revision']=='6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2'
env=record['build']['config']['python_environment']
assert env['factory']=='integrations.native_selfplay:NativeFrozenOpponentPufferEnvironment'
assert env['spec']['agents']==4096 and env['spec']['observation_size']==4851
assert env['spec']['action_sizes']==[3529] and not env['spec']['teacher']
assert env['options']['teacher'] is None and not env['options']['supervise_teacher']
run=copy.deepcopy(record['config'])
assert run['overrides']['train.gamma']==env['options']['shaping_gamma']==0.999
assert run['overrides']['train.learning_rate']==0.0015
assert run['overrides']['train.norm_adv']==1 and run['overrides']['train.replay_ratio']==1
assert run['overrides']['policy.hidden_size']==128 and run['overrides']['policy.num_layers']==4
assert run['overrides']['train.horizon']==128 and run['overrides']['train.minibatch_size']==8192
run['seed']=743
run['total_timesteps']=300_000_000
run['overrides']['base.checkpoint_interval']=128
run['initialize']={
 'run':'/recovery/classic-native-flat-frozen-gen1-pilot-29937/run',
 'checkpoint':'/recovery/classic-native-flat-frozen-gen1-pilot-29937/run/checkpoints/metta_generals/run/0000000033554432.bin',
 'sha256':'3ab08100c02dc06a3ffa3888e26487dc87df2d41408c43ae854ac47088d6d3af',
}
# Verified weight initialization; fresh optimizer/clocks, not an exact learner resume.
Path(sys.argv[2]).write_text(json.dumps(run)+'\n')
PYCONFIG
ln -s "$out/run" "$disk/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits >> "$out/gpu-timestamped.csv"
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-native-flat-frozen-gen1-300m-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 35m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/config.json" \
    --build "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/build" \
    --output "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$source/monitor_coworld_steady_interval.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-native-flat-frozen-gen1-300m 600 524288 30000 8 12 20; then
        echo 'Native self-play pilot stopped by throughput or progress guard' >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$source/monitor_coworld_steady_interval.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-native-flat-frozen-gen1-300m 600 524288 30000 8 12 20
checkpoint=$out/run/checkpoints/metta_generals/run/0000000299892736.bin
digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
timeout -k 30s 3m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/staged/verify_native_puffer_policy.py" \
    --source "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/build/source" \
    --build "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/build/build.json" \
    --training "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/run/training.json" \
    --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" --public-views \
    --output "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/native-parity" > "$out/native-parity.log" 2>&1
for step in 67108864 134217728 299892736; do
    for opponent in expander_harvester sentinel; do
        checkpoint=$(printf '%s/run/checkpoints/metta_generals/run/%016d.bin' "$out" "$step")
        digest=$(sha256sum "$checkpoint" | cut -d' ' -f1)
        tag=$step-$opponent
        timeout -k 30s 3m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
            python "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/staged/evaluate_coworld_frozen_greedy.py" --native \
            --build "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/build/build.json" \
            --run "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/run" \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" \
            --output "/recovery/classic-native-flat-frozen-gen1-300m-pilot-$SLURM_JOB_ID/$tag" > "$out/$tag.log" 2>&1
        python3 - "$out/$tag/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]))
print(r['checkpoint_sha256'],r['opponent'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
    done
done
NODE

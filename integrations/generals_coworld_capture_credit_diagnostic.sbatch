#!/usr/bin/env bash
# Replace __SOURCE_BUNDLE__ with pinned source files, including puffer_advantage_normalization.cuh.
# Diagnostic only: one-step real-game capture, native PPO and actual legal masks.
#SBATCH --job-name=relh-classic-capture-credit-pilot
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-capture-credit-pilot-%j.log
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
old=$disk/classic-flat-source-route-selfplay-pilot-29162
parent=$disk/classic-flat-scripted-potential-pilot-29432
out=$disk/classic-capture-credit-pilot-$SLURM_JOB_ID
mkdir -p "$out/staged"
base64 -d <<'BUNDLE' | tar -xz -C "$out/staged"
__SOURCE_BUNDLE__
BUNDLE
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
name=relh-classic-capture-credit-pilot-$SLURM_JOB_ID
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
    -e METTA_AUDIT_ACTION_MASK=1
    -e PYTHONPATH=/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/staged:/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib)
test "$(docker run "${common[@]}" --name "$name" --entrypoint nvidia-smi \
    relh-generals-b300:20260923b --query-gpu=uuid --format=csv,noheader)" = "$gpu"
python3 - "$parent/run/training.json" "$out/build-config.json" "$out/config.json" <<'PYCONFIG'
import copy,json,sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text())
assert record['build']['model_sha256']=='c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d'
config=copy.deepcopy(record['build']['config'])
env=config['python_environment']
assert env['spec']['agents']==4096 and env['spec']['observation_size']==4851
assert env['spec']['action_sizes']==[3529] and not env['spec'].get('teacher',False)
env['factory']='audit_puffer_capture_credit:CaptureCreditEnvironment'
env['options']={'parallel_games':4096,'shaping_gamma':0.999}
env['assets']=[]
env['source_modules']=['generals','integrations.puffer_codec']
Path(sys.argv[2]).write_text(json.dumps(config)+'\n')
run=copy.deepcopy(record['config'])
assert run.get('initialize') is None and run['seed']==739
assert run['overrides']['train.gamma']==env['options']['shaping_gamma']==0.999
assert run['overrides']['train.replay_ratio']==0.5
run['overrides']['train.norm_adv']=0
run['total_timesteps']=16_777_216
run['overrides']['base.checkpoint_interval']=8
Path(sys.argv[3]).write_text(json.dumps(run)+'\n')
PYCONFIG
timeout -k 30s 5m docker run "${common[@]}" --name "$name" relh-generals-b300:20260923b \
    python "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" build \
    --config "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/build-config.json" \
    --output "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<'PYBUILD'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['model_sha256']=='c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d'
assert r['config']['python_environment']['factory']=='audit_puffer_capture_credit:CaptureCreditEnvironment'
assert r['config']['python_environment']['spec']['observation_size']==4851
print('capture_credit_model_sha256='+r['model_sha256'],flush=True)
PYBUILD
ln -s "$out/run" "$disk/classic-capture-credit-pilot-$SLURM_JOB_ID-0"
(
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' "$name" \
            >> "$out/cpu.csv" 2>/dev/null || true
        sleep 2
    done
) > "$disk/classic-capture-credit-gpu-$SLURM_JOB_ID.csv" 2>&1 &
sampler=$!
mkdir -p "$out/timing"
timeout -k 30s 8m docker run "${common[@]}" --name "$name" \
    -e METTA_EPOCH_TIMING_DIR="/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/timing" \
    relh-generals-b300:20260923b \
    python "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/staged/launch_classic_frozen_opponent_training.py" train \
    --config "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/config.json" \
    --build "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/build" \
    --output "/recovery/classic-capture-credit-pilot-$SLURM_JOB_ID/run" > "$out/train.log" 2>&1 &
pid=$!
start=$(date +%s)
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$source/monitor_coworld_steady_interval.py" \
        "$disk" "$SLURM_JOB_ID" "$((now-start))" classic-capture-credit 600 524288 30000 8 12 20; then
        echo 'Flat scripted-fresh pilot stopped by throughput or progress guard' >&2
        cleanup
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
test -f "$out/run/completed.json"
python3 "$source/monitor_coworld_steady_interval.py" \
    "$disk" "$SLURM_JOB_ID" "$(( $(date +%s)-start ))" classic-capture-credit 600 524288 30000 8 12 20
python3 - "$out/run" "$out/capture-credit-summary.json" <<'PYCREDIT'
from pathlib import Path
import json,sys
root=Path(sys.argv[1])
reports=list((root/'environments').rglob('capture-credit.jsonl'))
assert len(reports)==1
rows=[json.loads(line) for line in reports[0].read_text().splitlines()]
assert len(rows)>=4
first=rows[0]['capture_rate'];last=rows[-1]['capture_rate']
r={'scope':'one-step capture diagnostic; not arena policy qualification',
   'first_capture_rate':first,'last_capture_rate':last,'windows':len(rows),
   'improvement':last-first,'learning_demonstrated':last>=0.98 and last-first>=0.025}
Path(sys.argv[2]).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r),flush=True)
PYCREDIT
NODE

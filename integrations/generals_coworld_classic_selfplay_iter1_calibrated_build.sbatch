#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-build
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:12:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-build-%j.log
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
out=$disk/classic-selfplay-iter1-build-$SLURM_JOB_ID
mkdir -p "$out"
parent=$disk/classic-selfplay-teacher-h128-build-27857/build.json
printf '%s  %s\n' 68987ab92c702b9f0591b435a6b089ab8b06f21a7ea82c5fa8c87806625e8f90 "$parent" | sha256sum -c -
printf '%s  %s\n' a0261c5d81806b8ba2b4ed9675dda0dc51bd08c29109c6ef92dc3cb26c470886 "$disk/metta_puffer-selfplay-teacher-v2-20260927.py" | sha256sum -c -
python3 - "$parent" "$out/config.json" <<'PY'
import json,sys
from pathlib import Path
source,target=map(Path,sys.argv[1:])
manifest=json.loads(source.read_text())
config=manifest['config']
assert manifest['model_state_words']==22956
assert config['python_environment']['factory'].endswith(':BatchedGeneralsSelfPlayPufferEnvironment')
assert config['python_environment']['options']['parallel_games']==4096
assert config['fabric']['options']['hint_prior_strength']==0.5
assert config['python_environment']['options'].get('move_hint_scale',1)==1
assert config['python_environment']['options'].get('split_hint_scale',1)==1
config['python_environment']['options'].update(move_hint_scale=0.25,split_hint_scale=0.25)
config['fabric']['teacher']={'interpolation':'constant','phases':[
    {'agent_steps':0,'ppo_coefficient':1.0,'coefficient':0.0,'action_mix':0.0}
]}
target.write_text(json.dumps(config)+'\n')
PY
gpu=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
[[ "$gpu" == GPU-* ]]
name=relh-classic-selfplay-iter1-build-$SLURM_JOB_ID
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
timeout -k 30s 9m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v "$disk:/recovery" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -v "$disk/metta_puffer-selfplay-teacher-v2-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain \
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain \
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh build \
    --config "/recovery/classic-selfplay-iter1-build-$SLURM_JOB_ID/config.json" \
    --output "/recovery/classic-selfplay-iter1-build-$SLURM_JOB_ID/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());c=d['config']
assert d['model_state_words']==22956
assert c['python_environment']['options']['move_hint_scale']==0.25
assert c['python_environment']['options']['split_hint_scale']==0.25
phases=c['fabric']['teacher']['phases']
assert len(phases)==1
assert all(phases[0][key]==value for key,value in {
    'agent_steps':0,'ppo_coefficient':1.0,'coefficient':0.0,'action_mix':0.0,
}.items())
print(json.dumps({'model_sha256':d['model_sha256'],'model_state_words':d['model_state_words']}),flush=True)
PY
NODE

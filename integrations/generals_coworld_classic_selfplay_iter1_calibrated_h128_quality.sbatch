#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-selfplay-iter1-calibrated-h128-quality
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:35:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/classic-selfplay-iter1-calibrated-h128-quality-%j.log
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
job=$SLURM_JOB_ID
out=$disk/classic-selfplay-iter1-calibrated-h128-quality-$job
mkdir -p "$out"
printf '%s  %s
' 3117a9d770ccbda27e10f1091d943ce8393332d231e70e970f29f823257c7b89 "$disk/evaluate_coworld_spatial_mixed-frozen-v1-20260927.py" | sha256sum -c -
printf '%s  %s\n' 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
printf '%s  %s\n' 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
run=$disk/classic-selfplay-iter1-calibrated-h128-28074/run
build=$disk/classic-selfplay-iter1-build-28048/build/build.json
test -f "$run/completed.json"
python3 - "$run" "$build" "$out/identities.json" <<'PYINNER'
import hashlib,json,sys
from pathlib import Path
run,build,out=map(Path,sys.argv[1:]);record=json.loads((run/'training.json').read_text())
assert record['build']==json.loads(build.read_text())
identity={}
for tag,step in [('early',16777216),('final',33554432)]:
    checkpoint=run/f'checkpoints/metta_generals/run/{step:016d}.bin'
    assert checkpoint.is_file()
    identity[tag]=dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
out.write_text(json.dumps(identity)+'\n')
print(json.dumps(identity),flush=True)
PYINNER
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}; memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
name=relh-classic-selfplay-iter1-calibrated-h128-quality-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap 'exit 143' TERM
for tag in early final; do
    checkpoint=$(python3 - "$out/identities.json" "$tag" <<'PYREF'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]['checkpoint'])
PYREF
)
    digest=$(python3 - "$out/identities.json" "$tag" <<'PYREF'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]['sha256'])
PYREF
)
    for opponent in expander_harvester sentinel; do
        timeout -k 30s 5m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=8 \
            --ulimit core=0:0 --user "$(id -u):$(id -g)" \
            -v "$w:/work" -v "$disk:/recovery" -v "$out:/recovery-output" \
            -v "$w/source-spatial2/integrations/generals_fabric.py:/work/source-normalized-16245/integrations/generals_fabric.py:ro" \
            -v "$disk/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro" \
            -v "$disk/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro" \
            -v /var/tmp/relh-generals-jax-cache:/cache \
            -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
            -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
            -e XDG_CACHE_HOME=/recovery-output/cache \
            -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource \
            -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
            -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
            relh-generals-b300:20260923b python /recovery/evaluate_coworld_spatial_mixed-frozen-v1-20260927.py \
            --build /recovery/classic-selfplay-iter1-build-28048/build/build.json \
            --run /recovery/classic-selfplay-iter1-calibrated-h128-28074/run \
            --checkpoint "${checkpoint/$disk/\/recovery}" --sha256 "$digest" \
            --games 128 --pool-size 128 --seed 1386 --opponent "$opponent" --reward-diagnostics \
            --output "/recovery-output/$tag-$opponent" > "$out/$tag-$opponent.log" 2>&1
        python3 - "$out/$tag-$opponent/evaluation.json" <<'PYSCORE'
import json,sys
r=json.load(open(sys.argv[1]));print(r['opponent'],r['checkpoint_sha256'],r['wins'],r['losses'],r['draws'],r['score'],flush=True)
PYSCORE
    done
done
NODE

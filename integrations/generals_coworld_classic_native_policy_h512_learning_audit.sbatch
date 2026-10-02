#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-h512-learning-audit
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-h512-learning-audit-%j.log
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


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" a7291d2a2f25166941b07309789865822e85ccbef721edbb199bc8fb3520e3ba \
    "$disk/metta_puffer-reward-scaled-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-h512-learning-audit-$job
mkdir -p "$out"
printf "%s  %s\n" a0ae63e5ca54953ecbf26324ff94efaeeab2cfc36ef160ef2ab8d885ddb51d0e "$disk/native_puffer_policy-v2-20260927.py" | sha256sum -c -
printf "%s  %s\n" 828d2d10c91b6de2522ffd3944faf9e587afa762c3fa4aaea2e87206f140007d "$disk/audit_native_action_distributions-20260927.py" | sha256sum -c -
python3 - "$out/config.json" <<"PY"
import json,sys
from pathlib import Path
root="/recovery"
build=root+"/device-context-native-policy-65536-build-25526/build/build.json"
refs={}
for name,run,step,digest,policy_build in (
 ("baseline","native-policy-initialized-512-probe-25880",33554432,"61dfb5f6bb06ec1e409f4ae8250d51872be7ff5004900a48deb84b12e52b1a08",build),
 ("low_lr_2b","native-policy-initialized-512-low-lr-5b-26068",2147483648,"59c150c77011d3dfc0c0ce36b770411c21093e3e46258c0d3623f20e425a2ea4",build),
 ("scaled_33m","native-policy-reward-scaled-33m-probe-26286",33554432,"d63dd53c54a7446f78f404e13fe32ab464e447556899f14b931ac4212df9d0c9",root+"/native-policy-reward-scaled-build-26260/build/build.json"),
):
 base=root+"/"+run+"/run"
 refs[name]=dict(build=policy_build,training=base+"/training.json",checkpoint=base+"/checkpoints/metta_generals/run/"+f"{step:016d}.bin",sha256=digest)
Path(sys.argv[1]).write_text(json.dumps(dict(policies=refs,environment_build=build,seed=1377,games=64,turns=192,parent="baseline"))+"\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
name=relh-classic-native-policy-h512-learning-audit-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v "$disk/native_puffer_policy-v2-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-reward-scaled-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/audit_native_action_distributions-20260927.py \
    --config /recovery-output/config.json --output /recovery-output/proof > "$out/container.log" 2>&1
cat "$out/proof/audit.json"
'

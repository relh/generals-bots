#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-initialized-512-low-lr-268m-validation
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-initialized-512-low-lr-268m-validation-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
disk=/var/tmp/relh-generals-recovery
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e75f1862f88847c8893344c8ff26ca0bb35ff284d085ff95b1b000f1c7833d75 \
    "$disk/metta_puffer-balanced-20260927.py" | sha256sum -c -
printf "%s  %s\n" 020dbafc66861496295787330361b07e0492c892805594105fdbb050dcbb37f5 \
    "$disk/puffer_codec-calibrated-v1.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/native-policy-initialized-512-low-lr-268m-validation-$job
mkdir -p "$out"
printf "%s  %s\n" 2de03fae3d9510df635b39c7a0cfce2bd58940fbdcb9251dbbfd4c02c0d7c713 \
    "$disk/evaluate_coworld_frozen_greedy-native-20260927.py" | sha256sum -c -
printf "%s  %s\n" b3b46175fde1a332706834cc60166d44c307c6cdc5811e60e8c9c009328f20e8 \
    "$disk/launch_native_initialized_evaluation-20260927.py" | sha256sum -c -
printf "%s  %s\n" 37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda \
    "$disk/puffer-calibrated-transfer.py" | sha256sum -c -
printf "%s  %s\n" a0ae63e5ca54953ecbf26324ff94efaeeab2cfc36ef160ef2ab8d885ddb51d0e "$disk/native_puffer_policy-v2-20260927.py" | sha256sum -c -
printf "%s  %s\n" 0958d85285144647aea4d32fc60f08e3965d82d4cad83f7b184a58949cd63ec4 "$disk/puffer-native-initializer-20260927.py" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
ready=0
for attempt in {1..15}; do
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
printf "GPU_QUERY attempt=%s step_gpus=%s rows=%s\n" "$attempt" "$SLURM_STEP_GPUS" "$usage"
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
echo "gpu_idle_check=$attempt gpu=$gpu memory_mb=$memory utilization=$utilization step_gpus=$SLURM_STEP_GPUS"
if (( memory < 2048 && utilization < 20 )); then
    ready=1
    break
fi
sleep 2
done
(( ready == 1 ))
test -n "$gpu"
name=relh-classic-native-policy-initialized-512-low-lr-268m-validation-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
for tag in baseline candidate; do
    run=native-policy-initialized-512-probe-25880
    checkpoint=0000000033554432.bin
    if [[ "$tag" == candidate ]]; then
        run=native-policy-initialized-512-low-lr-probe-25994
        checkpoint=0000000268435456.bin
    fi
    relative_checkpoint="checkpoints/metta_generals/run/$checkpoint"
    digest=$(python3 - "$disk/$run/run" "$relative_checkpoint" <<"PY"
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1])
checkpoint=run/sys.argv[2]
if run.name == "run" and "low-lr-probe-25994" in str(run):
    completed=json.loads((run/"completed.json").read_text())
    assert completed["trained_timesteps"] == 268435456
    assert Path(completed["final_checkpoint"]).name == checkpoint.name
identity=json.loads(Path(str(checkpoint)+".learner.json").read_text())
policy_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert policy_sha == identity["policy_sha256"]
assert hashlib.sha256(Path(str(checkpoint)+".learner").read_bytes()).hexdigest() == identity["state_sha256"]
assert hashlib.sha256((run/"training.json").read_bytes()).hexdigest() == identity["run_sha256"]
print(policy_sha)
PY
    )
    printf "%s  %s\n" "$digest" "$disk/$run/run/$relative_checkpoint" | sha256sum -c -
    for opponent in expander_harvester sentinel strong_mixed; do
timeout -k 30s 9m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery" \
    -v "$disk/native_puffer_policy-v2-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-balanced-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b python /recovery/launch_native_initialized_evaluation-20260927.py \
    --native --build /recovery/device-context-native-policy-65536-build-25526/build/build.json \
    --run "/recovery/$run/run" --checkpoint "/recovery/$run/run/$relative_checkpoint" \
    --sha256 "$digest" --games 1024 --pool-size 1024 --seed 1358 --opponent "$opponent" \
    --output "/recovery-output/$tag-$opponent" > "$out/$tag-$opponent.log" 2>&1
        cat "$out/$tag-$opponent/evaluation.json"
    done
done

'

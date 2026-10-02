#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-native-policy-reward-scaled-build
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
#SBATCH --output=/var/tmp/relh-generals-recovery/native-policy-reward-scaled-build-%j.log
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
out=$disk/native-policy-reward-scaled-build-$job
mkdir -p "$out"
python3 - "$disk/device-context-native-policy-65536-build-25526/build/build.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())["config"]
assert config["fabric"] is None and not config["python_environment"]["spec"]["teacher"]
config["python_environment"]["options"].update(
    reward_scale=.5, army_shaping_weight=.5, land_shaping_weight=.3, castle_shaping_weight=0.)
assert config["python_environment"]["options"]["imitation_weight"] == 0.
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
PY
printf "%s  %s\n" f64fb8b3d0e8185678bacbe850a4d821ac8664debbbef86dfb5dab9e5f3ace30 "$disk/verify_puffer_reward_scale-20260927.py" | sha256sum -c -
printf "%s  %s\n" a0ae63e5ca54953ecbf26324ff94efaeeab2cfc36ef160ef2ab8d885ddb51d0e "$disk/native_puffer_policy-v2-20260927.py" | sha256sum -c -
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
name=relh-classic-native-policy-reward-scaled-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v "$disk:/recovery:ro" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$disk/metta_puffer-reward-scaled-20260927.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$disk/native_puffer_policy-v2-20260927.py:/work/source-normalized-16245/integrations/native_puffer_policy.py:ro" \
    -v "$disk/puffer_codec-calibrated-v1.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e XDG_CACHE_HOME=/recovery-output/cache -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache -e JAX_PLATFORMS=cuda,cpu -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e LD_LIBRARY_PATH=/work/runtime/nvidia/cu13/lib \
    relh-generals-b300:20260923b bash -c "
    bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
      --config /recovery-output/config.json --output /recovery-output/build &&
    python /recovery/verify_puffer_reward_scale-20260927.py \
      --build /recovery/device-context-native-policy-65536-build-25526/build/build.json \
      --training /recovery/native-policy-initialized-512-probe-25880/run/training.json \
      --checkpoint /recovery/native-policy-initialized-512-probe-25880/run/checkpoints/metta_generals/run/0000000033554432.bin \
      --sha256 61dfb5f6bb06ec1e409f4ae8250d51872be7ff5004900a48deb84b12e52b1a08 \
      --output /recovery-output/proof" > "$out/container.log" 2>&1
python3 - "$out/build/build.json" "$disk/device-context-native-policy-65536-build-25526/build/build.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["fabric"] is None
assert manifest["model_state_words"] == 0
assert manifest["model_sha256"] == ""
assert manifest["config"]["python_environment"]["options"]["move_hint_scale"] == .375
assert manifest["config"]["python_environment"]["options"]["split_hint_scale"] == .125
assert manifest["config"]["python_environment"]["spec"]["agents"] == 65536
assert manifest["config"]["python_environment"]["options"]["shaping_gamma"] == .999
assert manifest["config"]["python_environment"]["options"]["reward_scale"] == .5
assert manifest["environment_sha256"] != json.loads(Path(sys.argv[2]).read_text())["environment_sha256"]
print("NATIVE_POLICY_BUILD_OK", manifest["binary_sha256"], manifest["model_sha256"], flush=True)
PY
'

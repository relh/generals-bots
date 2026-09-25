#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-teacher-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/device-teacher-eval-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
run=$w/device-teacher-pilot-17100/run
build=$w/device-heldout-17085/build
config=$w/device-heldout-17085/eval-config.json
checkpoint=$run/checkpoints/metta_generals/run/0000000016777216.bin
out=$w/device-teacher-eval-$job
mkdir -p "$out"
test -s "$checkpoint"
printf "%s  %s\n" ee4d597cc5ad35965255478ec1a7bdc74adff2523af97e2d8521f235bb55f647 \
    "$w/metta-device-eval-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" e4e69fdfc88602c950e50547e1c0b6d5f8b16c2f35c7d93342e478c07fe3183f \
    "$w/metta_puffer-device-bridge.py" | sha256sum -c -
python3 - "$run/completed.json" "$run/training.json" "$build/build.json" "$config" <<"PY"
import json, sys
from pathlib import Path
completed, training, target, evaluation = (json.loads(Path(path).read_text()) for path in sys.argv[1:])
assert completed["trained_timesteps"] == 16_777_216
assert completed["final_checkpoint"] == "checkpoints/metta_generals/run/0000000016777216.bin"
assert training["build"]["model_sha256"] == target["model_sha256"]
assert training["build"]["model_state_words"] == target["model_state_words"]
assert target["config"]["python_environment"]["options"]["opponent"] == "expander_harvester"
assert evaluation["seeds"] == [1101] and evaluation["allow_environment_transfer"]
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
gpu=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then gpu=${allocated_gpus[$i]}; break; fi
done
test -n "$gpu"
name=relh-classic-device-teacher-eval-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 15m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-bridge.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e JAX_COMPILATION_CACHE_DIR=/cache \
    -e PYTHONPATH=/work/metta-device-eval-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/eval_puffer_gpu.sh \
    /work/device-teacher-pilot-17100/run /work/device-heldout-17085/build \
    "/work/device-teacher-eval-$job/eval" /work/device-heldout-17085/eval-config.json \
    /work/device-teacher-pilot-17100/run/checkpoints/metta_generals/run/0000000016777216.bin \
    > "$out/eval-container.log" 2>&1
python3 - "$checkpoint" "$out/eval/evaluation.json" <<"PY"
import hashlib, json, sys
from pathlib import Path
checkpoint = Path(sys.argv[1])
record = json.loads(Path(sys.argv[2]).read_text())
assert record["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert record["config"]["seeds"] == [1101]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True), flush=True)
PY
'

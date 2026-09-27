#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-spatial-hint2-update
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/spatial-hint2-update-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/spatial-hint2-update-$job
build=$disk/spatial-hint2-mixed-build-27340
name=relh-classic-spatial-hint2-update-$job
mkdir -p "$out"
printf "%s  %s\n" 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 \
    "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" d254db7698d29c9947659925a15d68ad72fa0aad57bf89672348f0e51980ac38 \
    "/var/tmp/relh-generals-recovery/metta_puffer-land-gain-v1-20260927.py" | sha256sum -c -
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e \
    "$w/hybrid-periodic-build-13829/build.json" | sha256sum -c -
printf "%s  %s\n" 7ef182b2e978fadd3cbfe2792251ef9b50823d1c24b51c9141c36aa65ec95bdf \
    "$w/monitor_coworld_steady_interval.py" | sha256sum -c -
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
(( ${#allocated_gpus[@]} == ${#usage[@]} ))
chosen_gpu=
gpu_local=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then
        chosen_gpu=${allocated_gpus[$i]}
        gpu_local=$i
        break
    fi
done
test -n "$chosen_gpu"
echo "allocated_gpus=$SLURM_STEP_GPUS training_gpu=$chosen_gpu local_index=$gpu_local"
printf "%s  %s\n" 51bfc7d2ff13eb2a1ae2da46bee75f9c733a44ce9102731a47d2117db9c68c24 "$build/build.json" | sha256sum -c -
printf "%s  %s\n" 96fe65234bbc5c30119947c101276796c14da19b8d4f75219d9e4e5d2b90b8f4 "$build/puffer" | sha256sum -c -
parent=$disk/spatial-hint2-mixed-27340/run
printf "%s  %s\n" d9b8df6a2758ccdeb8cf562901bda4d7642dad378a3f4bfe2c86235dd604737a "$parent/checkpoints/metta_generals/run/0000000008388608.bin" | sha256sum -c -
python3 - "$disk/spatial-hint2-mixed-27340/config.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text())
record["total_timesteps"]=16_777_216
record["seed"]=692
record["initialize"]={
    "run": "/recovery/spatial-hint2-mixed-27340/run",
    "checkpoint": "/recovery/spatial-hint2-mixed-27340/run/checkpoints/metta_generals/run/0000000008388608.bin",
    "sha256": "d9b8df6a2758ccdeb8cf562901bda4d7642dad378a3f4bfe2c86235dd604737a",
    "allow_environment_transfer": True,
}
record["overrides"].update({
    "train.learning_rate": 0.003,
    "train.replay_ratio": 0.5,
    "train.gae_lambda": 0.99,
    "train.ent_coef": 0.003,
})
assert record["overrides"]["train.gamma"] == 0.999
Path(sys.argv[2]).write_text(json.dumps(record)+"\n")
PY
common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$w:/work" -v "$disk:/recovery"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -v /var/tmp/relh-generals-recovery/metta_puffer-land-gain-v1-20260927.py:/work/source-spatial2/integrations/metta_puffer.py:ro
    -v /var/tmp/relh-generals-recovery/puffer_codec-calibrated-v1.py:/work/source-spatial2/integrations/puffer_codec.py:ro
    -e HOME=/recovery -e XDG_CACHE_HOME=/recovery/cache-spatial-land-gain
    -e JAX_COMPILATION_CACHE_DIR=/recovery/jax-compile-cache-spatial-land-gain
    -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-spatial2:/work/oldsource)
cleanup() {
    kill "${sampler:-}" 2>/dev/null || true
    docker stop -t 5 "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap "exit 143" TERM
python3 - "$build/build.json" <<"PY"
import json, sys
d=json.load(open(sys.argv[1]))
assert d["config"]["environment_backend"] == "cuda"
assert d["config"]["python_environment"]["device_resident"] is True
assert d["config"]["python_environment"]["spec"]["agents"] == 8192
assert d["config"]["fabric"]["losses"] == []
assert d["config"]["fabric"]["factory"].endswith(":two_stage_tied_local_action_policy")
print("model_sha256="+d["model_sha256"])
PY
ln -s "$out/run" "$disk/spatial-hint2-update-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/spatial-hint2-update-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 16m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh train \
    --config "/recovery/spatial-hint2-update-$job/config.json" \
    --build "/recovery/spatial-hint2-mixed-build-27340" \
    --output "/recovery/spatial-hint2-update-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$w/monitor_coworld_steady_interval.py" \
        "$disk" "$job" "$((now-start))" spatial-hint2-update 600 262144 30000; then
        echo "PPO continuation stopped by throughput or progress guard" >&2
        status=1
        cleanup
        break
    fi
done
wait "$pid" || status=1
test -f "$out/run/completed.json" || status=1
exit "$status"
'

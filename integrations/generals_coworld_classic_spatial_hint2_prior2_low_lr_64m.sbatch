#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-spatial-hint2-prior2-low-lr
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/spatial-hint2-prior2-low-lr-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
disk=/var/tmp/relh-generals-recovery
job=$SLURM_JOB_ID
out=$disk/spatial-hint2-prior2-low-lr-$job
build=$disk/spatial-hint2-prior2-build-27432
name=relh-classic-spatial-hint2-prior2-low-lr-$job
mkdir -p "$out"
printf "%s  %s\n" 04d317499de676eb74a91deb2e8b52528c97831d5895a0e1f80b991426238992 \
    "$w/source-spatial2/integrations/generals_fabric.py" | sha256sum -c -
printf "%s  %s\n" d254db7698d29c9947659925a15d68ad72fa0aad57bf89672348f0e51980ac38 \
    "/var/tmp/relh-generals-recovery/metta_puffer-land-gain-v1-20260927.py" | sha256sum -c -
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e \
    "$w/hybrid-periodic-build-13829/build.json" | sha256sum -c -
printf "%s  %s\n" 9b88eb675c9781f4f1a4b670577cf61ce490622a89d22f16321f3d0213f028e0 \
    "$disk/monitor_coworld_steady_interval-v2.py" | sha256sum -c -
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
printf "%s  %s\n" d0702106f7f4a4c57ee9c446602b24fc12e1305fbca4af5409d62dd810ee0958 "$build/build.json" | sha256sum -c -
printf "%s  %s\n" 85d6aa7af46677a39390d64af77b7e8776727e4123d4ce28dba9438aaf28bcc8 "$build/puffer" | sha256sum -c -
parent=$disk/spatial-hint2-prior2-27432/run
printf "%s  %s\n" 0702a65083994e0160d0b07656ca9f29694926a02f498b485902ca55f6d650ef "$parent/checkpoints/metta_generals/run/0000000004194304.bin" | sha256sum -c -
python3 - "$disk/spatial-hint2-prior2-27432/config.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
record=json.loads(Path(sys.argv[1]).read_text())
record["total_timesteps"]=67_108_864
record["seed"]=694
record["initialize"]={
    "run": "/recovery/spatial-hint2-prior2-27432/run",
    "checkpoint": "/recovery/spatial-hint2-prior2-27432/run/checkpoints/metta_generals/run/0000000004194304.bin",
    "sha256": "0702a65083994e0160d0b07656ca9f29694926a02f498b485902ca55f6d650ef",
    "allow_environment_transfer": True,
}
record["overrides"].update({
    "train.learning_rate": 0.0003,
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
ln -s "$out/run" "$disk/spatial-hint2-prior2-low-lr-pilot-$job-0"
(
    while true; do
        date -u +%H:%M:%S
        nvidia-smi -i "$gpu_local" --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits
        sleep 1
    done
) > "$disk/spatial-hint2-prior2-low-lr-gpu-$job.csv" 2>&1 &
sampler=$!
timeout -k 30s 17m docker run "${common[@]}" --name "$name" \
    -e JAX_LOG_COMPILES=1 -e PYTHONFAULTHANDLER=1 \
    relh-generals-b300:20260923b bash /work/source-spatial2/integrations/run_puffer_gpu.sh train \
    --config "/recovery/spatial-hint2-prior2-low-lr-$job/config.json" \
    --build "/recovery/spatial-hint2-prior2-build-27432" \
    --output "/recovery/spatial-hint2-prior2-low-lr-$job/run" > "$out/container.log" 2>&1 &
pid=$!
start=$(date +%s)
status=0
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if ! python3 "$disk/monitor_coworld_steady_interval-v2.py" \
        "$disk" "$job" "$((now-start))" spatial-hint2-prior2-low-lr 600 262144 30000; then
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

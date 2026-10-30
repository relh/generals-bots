#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-mlp-dagger-two-30m
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/mlp-dagger-two-30m-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    source_root="$workspace/source-gpu"
    printf "%s\n" \
        "b7af348e03fa62ba40fbeb9f957037c57906503176d35423849375e2315e6de9  $source_root/metta_training/native_fabric.py" \
        "8843e97b107cfc9c6e2bae895ed539ad26d8f6cdc88b40bda891508c34859151  $source_root/integrations/generals_fabric.py" \
        "86a89f8d1c1d658e3eba932a1a965c3cc9b3495cdb8a1833cc72251028bfb1ba  $source_root/integrations/metta_puffer.py" \
        "2daa71db8a005406756c79f13c52b2831ce84bff3b4e9384a032edf6caa08452  $workspace/throughput-mlp-dagger4m-build.json" \
        "2942ec364d0c775e9f58427c0b0ed3226ad640fcb0fdfd5316fb6aa1a1967700  $workspace/throughput-mlp-dagger4m-train.json" | sha256sum --check
    nvidia-smi -L
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
build = json.loads((workspace / "throughput-mlp-dagger4m-build.json").read_text())
assert build["python_environment"]["options"]["classic_maps"] is True
assert build["python_environment"]["options"]["parallel_games"] == 1024
assert build["fabric"]["factory"] == "integrations.generals_fabric:memoryless_mlp_policy"
assert all(phase["action_mix"] == 0.5 for phase in build["fabric"]["teacher"]["phases"])
(workspace / "mlp-dagger-two-30m-build.json").write_text(json.dumps(build))
source_train = json.loads((workspace / "throughput-mlp-dagger4m-train.json").read_text())
for index in range(2):
    train = json.loads(json.dumps(source_train))
    train["seed"] = 341 + index
    train["total_timesteps"] = 31_457_280
    train["overrides"].update({
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 8192,
        "train.horizon": 32,
        "train.learning_rate": 0.0003,
        "base.checkpoint_interval": 16,
    })
    (workspace / f"mlp-dagger-two-30m-{index}.json").write_text(json.dumps(train))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/mlp-dagger-two-30m-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-mlp-dagger-two-30m-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 41m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=24 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-gpu/integrations/run_puffer_gpu.sh build \\
                --config /work/mlp-dagger-two-30m-build.json \\
                --output /work/build-mlp-dagger-two-30m-${SLURM_JOB_ID}
            pids=()
            for index in 0 1; do
                bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                    --config /work/mlp-dagger-two-30m-\${index}.json \\
                    --build /work/build-mlp-dagger-two-30m-${SLURM_JOB_ID} \\
                    --output /work/mlp-dagger-two-30m-${SLURM_JOB_ID}-\${index} \\
                    > /work/mlp-dagger-two-30m-${SLURM_JOB_ID}-\${index}.launch.log 2>&1 &
                pids+=(\$!)
            done
            for pid in \${pids[@]}; do wait \$pid; done
        " &
    runner=$!
    monitor_since=0
    while kill -0 "$runner" 2>/dev/null; do
        sleep 15
        if grep -Eq "NonFiniteGradsError|FloatingPointError" "$workspace"/mlp-dagger-two-30m-"${SLURM_JOB_ID}"-{0,1}/console.log 2>/dev/null; then
            echo "Nonfinite Fabric gradient; releasing GPU and preserving checkpoints." >&2
            docker stop -t 5 "$name" >/dev/null 2>&1 || true
            wait "$runner" || true
            exit 1
        fi
        if [[ -f "$workspace/mlp-dagger-two-30m-${SLURM_JOB_ID}-0/monitor-v2.json" ]]; then
            if (( monitor_since == 0 )); then monitor_since=$(date +%s); fi
            read -r minimum_epoch aggregate_sps completed gpu_mean < <(python3 - "$workspace" "$SLURM_JOB_ID" <<"PY"
import re
import statistics
import sys
from pathlib import Path

workspace, job = Path(sys.argv[1]), sys.argv[2]
epochs = []
rates = []
completed = 0
for index in range(2):
    path = workspace / f"mlp-dagger-two-30m-{job}-{index}"
    completed += (path / "completed.json").exists()
    rows = []
    if (path / "console.log").exists():
        for block in (path / "console.log").read_text().split("╭"):
            epoch = re.search(r"Epoch\s+(\d+)", block)
            sps = re.search(r"SPS\s+([\d.]+)([KM]?)", block)
            if epoch and sps:
                value = float(sps[1]) * {"": 1, "K": 1000, "M": 1_000_000}[sps[2]]
                rows.append((int(epoch[1]), value))
    epochs.append(rows[-1][0] if rows else 0)
    active = [value for epoch, value in rows if epoch > 1 and epoch % 8]
    rates.append(statistics.median(active[-5:]) if len(active) >= 5 else -1)
aggregate = sum(rates) if all(rate >= 0 for rate in rates) else -1
samples = []
for line in (workspace / f"mlp-dagger-two-30m-gpu-{job}.csv").read_text().splitlines():
    first = line.split(",", 1)[0].strip()
    if first.isdigit():
        samples.append(int(first))
gpu_mean = statistics.mean(samples[-60:]) if len(samples) >= 60 else -1
print(min(epochs), round(aggregate), completed, round(gpu_mean))
PY
            )
            if (( $(date +%s) - monitor_since >= 300 && minimum_epoch == 0 )); then
                echo "No completed epoch after 300 seconds; releasing GPU." >&2
                docker stop -t 5 "$name" >/dev/null 2>&1 || true
                wait "$runner" || true
                exit 1
            fi
            if (( completed == 0 && minimum_epoch >= 10 && aggregate_sps >= 0 && aggregate_sps < 30000 )); then
                echo "Aggregate throughput below 30000 SPS after warmup; releasing GPU." >&2
                docker stop -t 5 "$name" >/dev/null 2>&1 || true
                wait "$runner" || true
                exit 1
            fi
            if (( completed == 0 && minimum_epoch >= 10 && gpu_mean >= 0 && gpu_mean < 30 )); then
                echo "Last-minute GPU utilization below 30%; releasing GPU." >&2
                docker stop -t 5 "$name" >/dev/null 2>&1 || true
                wait "$runner" || true
                exit 1
            fi
        fi
    done
    wait "$runner"
'

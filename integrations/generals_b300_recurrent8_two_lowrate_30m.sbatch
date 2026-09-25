#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-recurrent8-two-lowrate-30m
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
#SBATCH --output=/tmp/relh-generals-gpu/recurrent8-two-lowrate-30m-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    source_root="$workspace/source-gpu"
    printf "%s\n" \
        "df3d13cdc922ddfa18dd236a8ffb59eee6be4b5c20dc7aeb0cae7ad0d068220b  $source_root/metta_training/fabric.py" \
        "dbdc1d59c7806c4ddf1e46b7631e7dc6cb8364ce2d3f3b28b77cf42bcf6437fd  $source_root/metta_training/model_config.py" \
        "38e744e327c220fe3860ebb505c6e69619eb84d3c482a54e7812157259d194b3  $source_root/metta_training/native_fabric.py" \
        "05365ed3224db4220cc0969be78651033e0917f8460eff5d70ffc0e7f4b84b8c  $source_root/metta_training/environment.py" \
        "86a89f8d1c1d658e3eba932a1a965c3cc9b3495cdb8a1833cc72251028bfb1ba  $source_root/integrations/metta_puffer.py" \
        "497b541266ba1875ec28a91f6a27c0b41efbfbbbf477bb0e7ddfaae15eac482e  $source_root/integrations/configs/gpu-batch-wide-build.json" \
        "d1e6c1e8788b908c1b97249c4d16e4698883b17392ab863edde88705f055151e  $source_root/integrations/configs/gpu-batch-30m.json" | sha256sum --check
    nvidia-smi -L
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
source = workspace / "source-gpu/integrations/configs"
build = json.loads((source / "gpu-batch-wide-build.json").read_text())
build["python_environment"]["options"]["parallel_games"] = 1024
build["python_environment"]["spec"]["agents"] = 1024
build["fabric"]["factory"] = "metta_training.fabric:recurrent_policy"
build["fabric"]["options"] = {"atoms": 8}
build["fabric"]["compiler"] = "standard"
(workspace / "recurrent8-two-lowrate-30m-build.json").write_text(json.dumps(build))
source_train = json.loads((source / "gpu-batch-30m.json").read_text())
for index in range(2):
    train = json.loads(json.dumps(source_train))
    train["seed"] = 261 + index
    train["overrides"].update({
        "vec.total_agents": 4096,
        "vec.num_threads": 4,
        "vec.num_buffers": 4,
        "train.minibatch_size": 8192,
        "train.horizon": 32,
        "train.learning_rate": 0.0001,
        "base.checkpoint_interval": 16,
    })
    (workspace / f"recurrent8-two-lowrate-30m-{index}.json").write_text(json.dumps(train))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/recurrent8-two-lowrate-30m-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-recurrent8-two-lowrate-30m-${SLURM_JOB_ID}"
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
                --config /work/recurrent8-two-lowrate-30m-build.json \\
                --output /work/build-recurrent8-two-lowrate-30m-${SLURM_JOB_ID}
            pids=()
            for index in 0 1; do
                bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                    --config /work/recurrent8-two-lowrate-30m-\${index}.json \\
                    --build /work/build-recurrent8-two-lowrate-30m-${SLURM_JOB_ID} \\
                    --output /work/recurrent8-two-lowrate-30m-${SLURM_JOB_ID}-\${index} \\
                    > /work/recurrent8-two-lowrate-30m-${SLURM_JOB_ID}-\${index}.launch.log 2>&1 &
                pids+=(\$!)
            done
            for pid in \${pids[@]}; do wait \$pid; done
        " &
    runner=$!
    monitor_since=0
    while kill -0 "$runner" 2>/dev/null; do
        sleep 15
        if grep -q "NonFiniteGradsError" "$workspace"/recurrent8-two-lowrate-30m-"${SLURM_JOB_ID}"-{0,1}/console.log 2>/dev/null; then
            echo "Nonfinite recurrent gradient; releasing GPU and preserving checkpoints." >&2
            docker stop -t 5 "$name" >/dev/null 2>&1 || true
            wait "$runner" || true
            exit 1
        fi
        if [[ -f "$workspace/recurrent8-two-lowrate-30m-${SLURM_JOB_ID}-0/monitor-v2.json" ]]; then
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
    path = workspace / f"recurrent8-two-lowrate-30m-{job}-{index}"
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
for line in (workspace / f"recurrent8-two-lowrate-30m-gpu-{job}.csv").read_text().splitlines():
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

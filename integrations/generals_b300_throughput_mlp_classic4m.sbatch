#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-throughput-mlp-classic4m
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/throughput-mlp-classic4m-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    test -s "$workspace/source-gpu/integrations/generals_fabric.py"
    nvidia-smi -L
    python3 - <<"PY"
import json
from pathlib import Path

workspace = Path("/tmp/relh-generals-gpu")
build = json.loads((workspace / "generals-wide16-build.json").read_text())
build["python_environment"]["options"]["parallel_games"] = 1024
build["python_environment"]["options"]["classic_maps"] = True
build["python_environment"]["options"]["horizon"] = 800
build["python_environment"]["spec"]["agents"] = 1024
build["fabric"]["factory"] = "integrations.generals_fabric:memoryless_mlp_policy"
build["fabric"]["options"] = {"hidden": 64}
build["fabric"]["compiler"] = "standard"
(workspace / "throughput-mlp-classic4m-build.json").write_text(json.dumps(build))
train = json.loads((workspace / "gpu-batch-pilot.json").read_text())
train["total_timesteps"] = 4194304
train["seed"] = 291
train["overrides"].update({
    "vec.total_agents": 4096,
    "vec.num_threads": 4,
    "vec.num_buffers": 4,
    "train.minibatch_size": 8192,
    "base.checkpoint_interval": 8,
})
(workspace / "throughput-mlp-classic4m-train.json").write_text(json.dumps(train))
PY
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/throughput-mlp-classic4m-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "relh-generals-throughput-mlp-classic4m-${SLURM_JOB_ID}" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 10m docker run --rm --ulimit core=0:0 --name "relh-generals-throughput-mlp-classic4m-${SLURM_JOB_ID}" \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b \
        bash -lc "
            set -euo pipefail
            bash /work/source-gpu/integrations/run_puffer_gpu.sh build \\
                --config /work/throughput-mlp-classic4m-build.json \\
                --output /work/build-throughput-mlp-classic4m-${SLURM_JOB_ID}
            bash /work/source-gpu/integrations/run_puffer_gpu.sh train \\
                --config /work/throughput-mlp-classic4m-train.json \\
                --build /work/build-throughput-mlp-classic4m-${SLURM_JOB_ID} \\
                --output /work/throughput-mlp-classic4m-${SLURM_JOB_ID}
        " &
    runner=$!
    monitor_since=0
    monitor="$workspace/throughput-mlp-classic4m-${SLURM_JOB_ID}/monitor-v2.json"
    console="$workspace/throughput-mlp-classic4m-${SLURM_JOB_ID}/console.log"
    while kill -0 "$runner" 2>/dev/null; do
        sleep 15
        if [[ -f "$monitor" ]]; then
            if (( monitor_since == 0 )); then monitor_since=$(date +%s); fi
            read -r steps idle < <(python3 - "$monitor" "$workspace/throughput-mlp-classic4m-gpu-${SLURM_JOB_ID}.csv" <<"PY"
import json
import statistics
import sys
from pathlib import Path

monitor = json.loads(Path(sys.argv[1]).read_text())
samples = []
for line in Path(sys.argv[2]).read_text().splitlines():
    first = line.split(",", 1)[0].strip()
    if first.isdigit():
        samples.append(int(first))
utilization = statistics.median(samples[-12:]) if samples else 0
print(int(monitor["step"]), int(utilization < 10))
PY
            )
            if (( $(date +%s) - monitor_since >= 240 && steps == 0 && idle == 1 )) &&
                ! grep -q "Steps" "$console" 2>/dev/null; then
                printf "No training steps after 240 seconds with median GPU utilization below 10%%; stopping smoke.\n" >&2
                docker stop -t 5 "relh-generals-throughput-mlp-classic4m-${SLURM_JOB_ID}" >/dev/null 2>&1 || true
                wait "$runner" || true
                exit 1
            fi
        fi
    done
    wait "$runner"
'

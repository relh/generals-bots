#!/usr/bin/env bash
#SBATCH --job-name=relh-generals-classic-teacher
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:07:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-gpu/classic-teacher-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --cpu-bind=cores bash -lc '
    set -euo pipefail
    workspace=/tmp/relh-generals-gpu
    (
        while true; do
            date -u +%H:%M:%S
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader,nounits || true
            sleep 1
        done
    ) > "$workspace/classic-teacher-gpu-${SLURM_JOB_ID}.csv" 2>&1 &
    sampler=$!
    name="relh-generals-classic-teacher-${SLURM_JOB_ID}"
    cleanup() {
        kill "$sampler" 2>/dev/null || true
        docker stop -t 5 "$name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 6m docker run -i --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=8 \
        --user "$(id -u):$(id -g)" -v "$workspace:/work" -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-gpu \
        relh-generals-b300:20260923b python3 -u - <<"PY"
import json
import time
from pathlib import Path

import jax
import numpy as np

from generals.agents.harvester_agent import HarvesterAgent
from generals.core import game
from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from metta_training.environment import EnvironmentContext

assert jax.devices("cuda"), "Classic teacher benchmark requires a CUDA GPU"
teacher = HarvesterAgent()
act = jax.jit(jax.vmap(lambda state, side, key: teacher.act(game.get_observation(state, side), key)))
results = []
for seed in (901, 902):
    context = EnvironmentContext(seed=seed, index=0, mode="evaluate", output=Path("/work"))
    env = BatchedGeneralsPufferEnvironment(
        context=context, board_size=10, horizon=800, opponent="mixed",
        shaping_weight=1.0, teacher=None, supervise_teacher=False,
        factorized_actions=True, classic_maps=True, parallel_games=1024,
    )
    env.reset(f"{seed}:0:0")
    started = time.monotonic()
    for turn in range(800):
        actions = np.asarray(act(env.states, env.sides, env.keys))
        indices = np.where(actions[:, 0] == 1, 400, actions[:, 3] * 100 + actions[:, 1] * 10 + actions[:, 2])
        transition = env.step(np.stack((indices, actions[:, 4]), axis=1).tolist())
        if (turn + 1) % 100 == 0:
            print(f"seed={seed} turn={turn + 1} elapsed={time.monotonic() - started:.1f}s", flush=True)
        if transition.episode_done:
            break
    result = {"seed": seed, "parallel_games": 1024, "turns": turn + 1,
              "score": transition.score, "perf": transition.perf,
              "elapsed_seconds": time.monotonic() - started}
    results.append(result)
    print(json.dumps(result), flush=True)
    env.close()
Path("/work/classic-teacher-results.json").write_text(json.dumps(results, indent=2))
PY
'

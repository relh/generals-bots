#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-strong-paired-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/strong-paired-eval-%j.log
set -euo pipefail

opponent=${1:?Use expander_harvester or sentinel}
seed=${2:?Use a held-out integer seed}
case "$opponent" in
    expander_harvester|sentinel) ;;
    *) echo "Unsupported opponent: $opponent" >&2; exit 2 ;;
esac
workspace=/tmp/relh-generals-coworld
source_build="$workspace/expander-quad-stability-build-13219"
run="$workspace/buffer4-long-13316/run"
config="$workspace/strong-${opponent}-build-${SLURM_JOB_ID}.json"
eval_config="$workspace/strong-${opponent}-eval-${SLURM_JOB_ID}.json"
build="$workspace/strong-${opponent}-build-${SLURM_JOB_ID}"
eval_name="relh-classic-strong-${opponent}-${SLURM_JOB_ID}"

python3 - "$source_build" "$run" "$config" "$eval_config" "$opponent" "$seed" <<'PY'
import json
import sys
from pathlib import Path

source_build, run, config, eval_config = map(Path, sys.argv[1:5])
opponent = sys.argv[5]
seed = int(sys.argv[6])
manifest = json.loads((source_build / "build.json").read_text())
training = json.loads((run / "run.json").read_text())
if manifest != training["build"]:
    raise ValueError("Source build differs from the completed training build")
if json.loads((run / "completed.json").read_text())["trained_timesteps"] != 41_943_040:
    raise ValueError("Training run is incomplete")
target = manifest["config"]
environment = target["python_environment"]
options = environment["options"]
if not (options["coworld_classic"] and options["parallel_games"] == 1024
        and environment["spec"]["agents"] == 1024):
    raise ValueError("Expected 1,024-game Coworld Classic environment")
options["opponent"] = opponent
config.write_text(json.dumps(target) + "\n")
eval_config.write_text(json.dumps({"seeds": [seed], "episodes_per_seed": 1,
                                   "allow_environment_transfer": True,
                                   "timeout_seconds": 900}) + "\n")
PY

export workspace source_build run config eval_config build eval_name opponent seed
srun --export=ALL --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -c '
    set -euo pipefail
    name="${eval_name}-build"
    cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
    trap cleanup EXIT
    trap "exit 143" TERM
    timeout -k 30s 3m docker run --rm --name "$name" --ulimit core=0:0 \
        --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
        --user "$(id -u):$(id -g)" \
        -v "$workspace:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
        -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
        -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource \
        relh-generals-b300:20260923b bash /work/source/integrations/run_puffer_gpu.sh build \
            --config "/work/${config#"$workspace/"}" \
            --output "/work/${build#"$workspace/"}"
'

for step in 31457280; do
    padded=$(printf '%016d' "$step")
    checkpoint="$run/checkpoints/metta_generals/run/${padded}.bin"
    output="$workspace/strong-${opponent}-${step}-${SLURM_JOB_ID}"
    expected=939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499
    if (( step == 41943040 )); then
        expected=ef5f40c7d7c543a45befec19433a3f402a0ae1e087a671dca087388f22d36353
    fi
    printf '%s  %s\n' "$expected" "$checkpoint" | sha256sum -c -
    export checkpoint output expected step
    srun --export=ALL --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --cpu-bind=cores bash -c '
        set -euo pipefail
        name="${eval_name}-${step}"
        cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
        trap cleanup EXIT
        trap "exit 143" TERM
        timeout -k 30s 18m docker run --rm --name "$name" --ulimit core=0:0 \
            --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
            --user "$(id -u):$(id -g)" \
            -v "$workspace:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
            -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
            -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source:/work/oldsource \
            relh-generals-b300:20260923b bash /work/source/integrations/eval_puffer_gpu.sh \
                "/work/${run#"$workspace/"}" "/work/${build#"$workspace/"}" \
                "/work/${output#"$workspace/"}" "/work/${eval_config#"$workspace/"}" \
                "/work/${checkpoint#"$workspace/"}"
    '
    python3 - "$output" "$opponent" "$expected" "$seed" <<'PY'
import json
import sys
from pathlib import Path

output, opponent, expected, seed = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
record = json.loads((output / "evaluation.json").read_text())
assert record["checkpoint_sha256"] == expected
assert record["source_build"] is not None
assert record["build"]["config"]["python_environment"]["options"]["opponent"] == opponent
assert record["config"]["allow_environment_transfer"] is True
assert record["config"]["seeds"] == [seed]
assert len(record["results"]) == 1 and record["results"][0]["games"] >= 1
print(json.dumps({"checkpoint_sha256": expected, "opponent": opponent,
                  "seed": seed, "result": record["results"][0]}, sort_keys=True))
PY
done

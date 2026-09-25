#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-cardinal2-ppo-build
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/cardinal2-ppo-prebuild-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/cardinal2-ppo-prebuild-$job
name=relh-classic-cardinal2-ppo-build-$job
mkdir -p "$output"
printf "%s  %s\n" ce893e5cc45a81df06921dc638c18be4beadf70faec4f38177a8c286787cd9c0 \
    "$workspace/cardinal2-mb16-long-build-15207/build.json" | sha256sum -c -
printf "%s  %s\n" fb9f235c8c8283db19da7b63e2768f1a5820cf359672bbbd4c6c249f885feff0 \
    "$workspace/cardinal2-mb16-long-15207/run/checkpoints/metta_generals/run/0000000025165824.bin" | sha256sum -c -
printf "%s  %s\n" f3582f20dede5ccdfb84edb5652ce2a652dc1cd3ca942d23d35adb6c7ae3e2bd \
    "$workspace/source-general-distance/integrations/metta_puffer.py" | sha256sum -c -
python3 - "$workspace/cardinal2-mb16-long-build-15207/build.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
build = json.loads(source.read_text())
assert build["model_sha256"] == "ae3ab7e3e7454a99c2a7927a3e5af23bf4e46227dc7b9ab68d03c26283eb2fa5"
assert build["model_state_words"] == 14124
config = build["config"]
options = config["python_environment"]["options"]
assert options["coworld_classic"] and options["packed_context_hint_features"]
assert options["opponent"] == "expander_harvester"
options["opponent"] = "strong_mixed"
options["shaping_weight"] = 1.0
options["castle_shaping_weight"] = 1.0
losses = config["fabric"]["losses"]
assert len(losses) == 1 and losses[0]["name"] == "sparse_teacher"
assert losses[0]["replace_ppo"] and losses[0]["options"]["coefficient"] == 1.0
losses[0]["replace_ppo"] = False
losses[0]["options"]["coefficient"] = 0.25
target.write_text(json.dumps(config) + "\n")
PY
trap "docker stop -t 5 $name >/dev/null 2>&1 || true" EXIT
timeout -k 30s 6m docker run --rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" -v "$workspace:/work" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-general-distance:/work/oldsource \
    --name "$name" relh-generals-b300:20260923b \
    bash /work/source-general-distance/integrations/run_puffer_gpu.sh build \
    --config "/work/cardinal2-ppo-prebuild-$job/config.json" \
    --output "/work/cardinal2-ppo-prebuild-$job/build" > "$output/build.log" 2>&1
python3 - "$output/build/build.json" <<"PY"
import json
import sys
build = json.load(open(sys.argv[1]))
assert build["model_state_words"] == 14124
assert build["config"]["fabric"]["losses"][0]["replace_ppo"] is False
print("target_model_sha256=" + build["model_sha256"])
PY
'

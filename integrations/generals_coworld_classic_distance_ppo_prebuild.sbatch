#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-distance-ppo-build
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
#SBATCH --output=/tmp/relh-generals-coworld/distance-ppo-prebuild-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
output=$workspace/distance-ppo-prebuild-$job
name=relh-classic-distance-ppo-build-$job
mkdir -p "$output"
printf "%s  %s\n" 930d4a8cc7921b003de55a4dcc175d26444e766423000bebaa1350196e4b82d1 \
    "$workspace/packed-hint-long-build-14603/build.json" | sha256sum -c -
printf "%s  %s\n" bc5a3dc51ca41b709472f7a07a11dfb58be24ceef9d6f3b37fc77475ac103a31 \
    "$workspace/packed-hint-long-14603/run/checkpoints/metta_generals/run/0000000029360128.bin" | sha256sum -c -
printf "%s  %s\n" f3582f20dede5ccdfb84edb5652ce2a652dc1cd3ca942d23d35adb6c7ae3e2bd \
    "$workspace/source-general-distance/integrations/metta_puffer.py" | sha256sum -c -
python3 - "$workspace/packed-hint-long-build-14603/build.json" "$output/config.json" <<"PY"
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
build = json.loads(source.read_text())
assert build["model_sha256"] == "4bd5a428012dfa8c1d24f4a238a2350ab5927b4cc1dc3d6d8ae2e88922a16fa1"
assert build["model_state_words"] == 14124
config = build["config"]
options = config["python_environment"]["options"]
assert options["coworld_classic"] and options["packed_context_hint_features"]
assert options["opponent"] == "expander_harvester"
options["opponent"] = "strong_mixed"
assert options["packed_context_hint_features"] is True
options["packed_context_hint_features"] = False
options["general_distance_hint_features"] = True
options["shaping_weight"] = 1.0
options["castle_shaping_weight"] = 1.0
losses = config["fabric"]["losses"]
assert len(losses) == 1 and losses[0]["name"] == "sparse_teacher"
assert losses[0]["replace_ppo"] and losses[0]["options"]["coefficient"] == 1.0
losses[0]["replace_ppo"] = False
losses[0]["options"]["coefficient"] = 0.25
target.write_text(json.dumps(config) + "\n")
PY
trap "docker stop -t 5 $name-test $name >/dev/null 2>&1 || true" EXIT
timeout -k 30s 3m docker run --rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" -v "$workspace:/work" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-general-distance:/work/oldsource \
    --name "$name-test" relh-generals-b300:20260923b bash -lc \
    "cd /work/source-general-distance && python -m pytest tests/test_metta_puffer.py tests/test_coworld_neural_codec.py -q -k general_distance && python -m pytest tests/test_coworld_neural_codec.py -q -k wire_view" \
    > "$output/test.log" 2>&1
timeout -k 30s 6m docker run --rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" -v "$workspace:/work" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-general-distance:/work/oldsource \
    --name "$name" relh-generals-b300:20260923b \
    bash /work/source-general-distance/integrations/run_puffer_gpu.sh build \
    --config "/work/distance-ppo-prebuild-$job/config.json" \
    --output "/work/distance-ppo-prebuild-$job/build" > "$output/build.log" 2>&1
python3 - "$output/build/build.json" <<"PY"
import json
import sys
build = json.load(open(sys.argv[1]))
assert build["model_state_words"] == 14124
assert build["config"]["fabric"]["losses"][0]["replace_ppo"] is False
print("target_model_sha256=" + build["model_sha256"])
PY
'

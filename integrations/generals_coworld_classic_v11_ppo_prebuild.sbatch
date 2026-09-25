#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-v11-ppo-build
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/v11-ppo-build-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/v11-ppo-build-$job
mkdir -p "$out"
printf "%s  %s\n" c911cba82cd465977d6ea79a649df8516c93d6ac2d6500bd7017268043ee5c0e "$w/hybrid-periodic-build-13829/build.json" | sha256sum -c -
printf "%s  %s\n" 8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346 "$w/hybrid-clean-13916/run/checkpoints/metta_generals/run/0000000033554432.bin" | sha256sum -c -
printf "%s  %s\n" ae3f68b80edcee43dd2968eb438e2dbb6d7f156f4fae732079989f0b7c8f86cb "$w/source-hybrid-teacher/integrations/metta_puffer.py" | sha256sum -c -
python3 - "$w/hybrid-periodic-build-13829/build.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
build = json.loads(Path(sys.argv[1]).read_text())
assert build["model_sha256"] == "c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9"
assert build["model_state_words"] == 12360
config = build["config"]
options = config["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["sentinel_teacher_fraction"] == 1.0 and options["sentinel_teacher_interval"] == 4
options["opponent"] = "strong_mixed"
options["shaping_weight"] = 1.0
options["castle_shaping_weight"] = 1.0
losses = config["fabric"]["losses"]
assert len(losses) == 1 and losses[0]["name"] == "sparse_teacher"
assert losses[0]["replace_ppo"] and losses[0]["options"]["coefficient"] == 1.0
losses[0]["replace_ppo"] = False
losses[0]["options"]["coefficient"] = 0.25
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
PY
name=relh-classic-v11-ppo-build-$job
trap "docker stop -t 5 $name >/dev/null 2>&1 || true" EXIT
timeout -k 30s 6m docker run --rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=8 --ulimit core=0:0 \
  --user "$(id -u):$(id -g)" -v "$w:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
  -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work -e HOME=/work \
  -e PYTHONPATH=/work/runtime:/work/source-hybrid-teacher:/work/oldsource \
  --name "$name" relh-generals-b300:20260923b \
  bash /work/source-hybrid-teacher/integrations/run_puffer_gpu.sh build \
  --config "/work/v11-ppo-build-$job/config.json" --output "/work/v11-ppo-build-$job/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<"PY"
import json, sys
build = json.load(open(sys.argv[1]))
assert build["model_state_words"] == 12360
assert build["config"]["fabric"]["losses"][0]["replace_ppo"] is False
print("target_model_sha256=" + build["model_sha256"])
PY
'

#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-v11-lowteacher-build
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
#SBATCH --output=/tmp/relh-generals-coworld/v11-lowteacher-build-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
out=$w/v11-lowteacher-build-$job
mkdir -p "$out"
printf "%s  %s\n" 9e6bf0ef59b788800a0da187d1ab76ebb55c6ed1c5241110f86d10e37ab570a3 "$w/v11-ppo-build-15317/build.json" | sha256sum -c -
printf "%s  %s\n" 8f641eae8d3958b319f48fa8e4831b4010ec28b1e20f825653f07f9659441346 "$w/hybrid-clean-13916/run/checkpoints/metta_generals/run/0000000033554432.bin" | sha256sum -c -
printf "%s  %s\n" 459e10eab59671c160d11ae8f49a3ffa31e487adf3a606ac7fedc3748745a56d "$w/source-v11-strongmix/integrations/metta_puffer.py" | sha256sum -c -
python3 - "$w/v11-ppo-build-15317/build.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
build = json.loads(Path(sys.argv[1]).read_text())
assert build["model_sha256"] == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
assert build["model_state_words"] == 12360
config = build["config"]
options = config["python_environment"]["options"]
assert options["coworld_classic"] and options["parallel_games"] == 1024
assert options["sentinel_teacher_fraction"] == 1.0 and options["sentinel_teacher_interval"] == 4
assert options["opponent"] == "strong_mixed" and options["shaping_weight"] == 1.0
assert "castle_shaping_weight" not in options
losses = config["fabric"]["losses"]
assert len(losses) == 1 and losses[0]["name"] == "sparse_teacher"
assert losses[0]["replace_ppo"] is False and losses[0]["options"]["coefficient"] == 0.25
losses[0]["options"]["coefficient"] = 0.05
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
PY
name=relh-classic-v11-lowteacher-build-$job
trap "docker stop -t 5 $name >/dev/null 2>&1 || true" EXIT
timeout -k 30s 6m docker run --rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=8 --ulimit core=0:0 \
  --user "$(id -u):$(id -g)" -v "$w:/work" -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
  -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work -e HOME=/work \
  -e PYTHONPATH=/work/runtime:/work/source-v11-strongmix:/work/oldsource \
  --name "$name" relh-generals-b300:20260923b \
  bash /work/source-v11-strongmix/integrations/run_puffer_gpu.sh build \
  --config "/work/v11-lowteacher-build-$job/config.json" --output "/work/v11-lowteacher-build-$job/build" > "$out/build.log" 2>&1
python3 - "$out/build/build.json" <<"PY"
import json, sys
build = json.load(open(sys.argv[1]))
assert build["model_state_words"] == 12360
assert build["config"]["fabric"]["losses"][0]["replace_ppo"] is False
assert build["config"]["fabric"]["losses"][0]["options"]["coefficient"] == 0.05
print("target_model_sha256=" + build["model_sha256"])
PY
'

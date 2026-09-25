#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-nopass-eval
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:22:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/nopass-eval-%j.log
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: sbatch $0 OPPONENT SEED" >&2
    exit 2
fi
case "$1" in random|expander_harvester|sentinel) ;; *) exit 2 ;; esac
export NOPASS_OPPONENT=$1 NOPASS_SEED=$2

srun --export=ALL --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
source_dir=$workspace/source-nopass-16245
printf "%s  %s\n" 51d72aaf2502c3b96a48d815c362b9015ffb73d183b35b28cbe49954b5e1c5cf "$source_dir/integrations/metta_puffer.py" | sha256sum -c -
train_run=$workspace/small-to-classic-ppo-16245/run
source_build=$workspace/small-to-classic-ppo-build-16245/build.json
checkpoint=$train_run/checkpoints/metta_generals/run/0000000008388608.bin
printf "%s  %s\n" 71806c20b4e755bcc53abceff19a83db84306eb68f15478f051327a86656b3df "$checkpoint" | sha256sum -c -
opponent=$NOPASS_OPPONENT
seed=$NOPASS_SEED
output=$workspace/nopass-eval-16245-$opponent-$seed-$SLURM_JOB_ID
build=$workspace/nopass-build-16245-$opponent-$SLURM_JOB_ID
config=$workspace/nopass-eval-16245-$opponent-$seed-$SLURM_JOB_ID.json
python3 - "$source_build" "$output.build-config.json" "$config" "$opponent" "$seed" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["python_environment"]["options"]["opponent"] == "random"
assert manifest["config"]["python_environment"]["options"]["coworld_classic"]
assert manifest["config"]["fabric"]["action_sizes"] == [1765, 2]
build = manifest["config"]
build["python_environment"]["options"]["opponent"] = sys.argv[4]
build["python_environment"]["options"]["mask_pass_when_moves_exist"] = True
Path(sys.argv[2]).write_text(json.dumps(build) + "\n")
Path(sys.argv[3]).write_text(json.dumps({"seeds": [int(sys.argv[5])],
    "episodes_per_seed": 1, "allow_environment_transfer": True,
    "timeout_seconds": 900}) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
mapfile -t usage < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
chosen_gpu=
for i in "${!allocated_gpus[@]}"; do
    IFS=, read -r memory utilization <<< "${usage[$i]}"
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory < 2048 && utilization < 20 )); then
        chosen_gpu=${allocated_gpus[$i]}
        break
    fi
done
test -n "$chosen_gpu"
name=relh-classic-nopass-$SLURM_JOB_ID
docker_common=(--rm --gpus "device=${chosen_gpu}" --cpus=16 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-nopass-16245:/work/oldsource)
cleanup() { docker stop -t 5 "${name}-build" "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 3m docker run "${docker_common[@]}" --name "${name}-build" \
    relh-generals-b300:20260923b bash /work/source-nopass-16245/integrations/run_puffer_gpu.sh build \
    --config "/work/${output#"$workspace/"}.build-config.json" \
    --output "/work/${build#"$workspace/"}"
timeout -k 30s 17m docker run "${docker_common[@]}" --name "$name" \
    relh-generals-b300:20260923b bash /work/source-nopass-16245/integrations/eval_puffer_gpu.sh \
    "/work/${train_run#"$workspace/"}" "/work/${build#"$workspace/"}" \
    "/work/${output#"$workspace/"}" "/work/${config#"$workspace/"}" \
    "/work/${checkpoint#"$workspace/"}"
python3 - "$output" "$opponent" "$seed" <<"PY"
import json, sys
from pathlib import Path
record = json.loads((Path(sys.argv[1]) / "evaluation.json").read_text())
assert record["checkpoint_sha256"] == "71806c20b4e755bcc53abceff19a83db84306eb68f15478f051327a86656b3df"
assert record["build"]["config"]["python_environment"]["options"]["opponent"] == sys.argv[2]
assert record["build"]["config"]["python_environment"]["options"]["mask_pass_when_moves_exist"]
assert record["config"]["seeds"] == [int(sys.argv[3])]
assert record["config"]["allow_environment_transfer"]
assert len(record["results"]) == 1
print(json.dumps(record["results"][0], sort_keys=True))
PY
'

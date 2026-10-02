#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-device-context-teacher05-16384-build
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --nice=2147483645
#SBATCH --chdir=/tmp
#SBATCH --output=/var/tmp/relh-generals-recovery/device-context-teacher05-16384-build-%j.log
set -euo pipefail

# Mandatory user policy: verify lowest scheduling priority before workload startup.
slurm_task_priority_record=$(scontrol show job -o "${SLURM_JOB_ID:?}")
[[ " $slurm_task_priority_record " == *" Nice=2147483645 "* ]] || { printf 'Lowest Slurm Nice verification failed\n' >&2; exit 1; }
slurm_task_priority=unknown
for slurm_task_field in $slurm_task_priority_record; do
  case "$slurm_task_field" in Priority=*) slurm_task_priority=${slurm_task_field#Priority=} ;; esac
done
[[ "$slurm_task_priority" =~ ^[0-9]+$ ]] || exit 1
printf 'SLURM_PRIORITY_READBACK JobId=%s Nice=2147483645 Priority=%s\n' "$SLURM_JOB_ID" "$slurm_task_priority" >&2
unset slurm_task_priority_record slurm_task_field


srun --nice=2147483645 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 bash -c '
set -euo pipefail
w=/tmp/relh-generals-coworld
job=$SLURM_JOB_ID
printf "%s  %s\n" 24034ec915b7b91a8a3e9333ce12b3dcbd9e517c0eb7c828cdf500c297d2fc06 \
    "$w/metta-device-wide-source.tar.gz" | sha256sum -c -
printf "%s  %s\n" ea5106c8467f10465d39e6a2779cca58fee464f791e1f7f36665041a76357151 \
    "$w/metta_puffer-device-fused.py" | sha256sum -c -
printf "%s  %s\n" a38ca81c920091a43f444ea14a8c0decdff0eea2412ddda600f78472b872b799 \
    "$w/puffer_codec-hintreuse.py" | sha256sum -c -
disk=/var/tmp/relh-generals-recovery
out=$disk/device-context-teacher05-16384-build-$job
mkdir -p "$out"
python3 - "$disk/device-context-teacher05-build-23518/build/build.json" "$out/config.json" <<"PY"
import json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())["config"]
assert config["fabric"]["teacher"]["phases"][0]["coefficient"] == .05
config["python_environment"]["spec"]["agents"] = 16384
config["python_environment"]["options"]["parallel_games"] = 16384
Path(sys.argv[2]).write_text(json.dumps(config) + "\n")
PY
IFS=, read -r -a allocated_gpus <<< "$SLURM_STEP_GPUS"
test "${#allocated_gpus[@]}" -eq 1
usage=$(nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
mapfile -t rows <<< "$usage"
test "${#rows[@]}" -eq 1
IFS=, read -r gpu memory utilization <<< "${rows[0]}"
gpu=${gpu//[[:space:]]/}
memory=${memory//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ "$gpu" == GPU-* && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]]
(( memory < 2048 && utilization < 20 ))
test -n "$gpu"
name=relh-classic-device-context-teacher05-16384-build-$job
cleanup() { docker stop -t 5 "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
trap "exit 143" TERM
timeout -k 30s 12m docker run --rm --name "$name" --gpus "device=$gpu" --cpus=16 \
    --ulimit core=0:0 --user "$(id -u):$(id -g)" \
    -v "$w:/work" \
    -v "$out:/recovery-output" \
    -v /var/tmp/relh-generals-jax-cache:/cache \
    -v "$w/metta_puffer-device-fused.py:/work/source-normalized-16245/integrations/metta_puffer.py:ro" \
    -v "$w/puffer_codec-hintreuse.py:/work/source-normalized-16245/integrations/puffer_codec.py:ro" \
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro \
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work \
    -e HOME=/work -e PYTHONPATH=/work/metta-device-wide-source:/work/runtime:/work/source-normalized-16245:/work/oldsource \
    -e JAX_COMPILATION_CACHE_DIR=/cache \
    relh-generals-b300:20260923b bash /work/source-normalized-16245/integrations/run_puffer_gpu.sh build \
    --config /recovery-output/config.json \
    --output /recovery-output/build > "$out/container.log" 2>&1
python3 - "$out/build/build.json" <<"PY"
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
assert manifest["config"]["fabric"]["teacher"]["phases"][0]["coefficient"] == .05
assert manifest["model_state_words"] == 22956
assert manifest["model_sha256"] == "3f75f84ede08faa047817b7ddab83ff4298c279f43c78c6d22bbebb26a81088b"
assert manifest["config"]["python_environment"]["spec"]["agents"] == 16384
print("TEACHER05_BUILD_OK", manifest["binary_sha256"], manifest["model_sha256"], flush=True)
PY
'

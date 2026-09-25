#!/usr/bin/env bash
#SBATCH --job-name=relh-classic-scripted-quality
#SBATCH --partition=b300
#SBATCH --nodelist=metta-fabric-b300-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --nice=100
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/relh-generals-coworld/scripted-quality-%j.log
set -euo pipefail

srun --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 bash -c '
set -euo pipefail
workspace=/tmp/relh-generals-coworld
source_root=$workspace/source-scripted-benchmark
printf "%s  %s\n" 63231ba24d53a29fc636e400c1aa933eb2f3e6a23b3db124c45809344eac93ca \
    "$source_root/integrations/benchmark_coworld_scripted.py" | sha256sum -c -
output=$workspace/scripted-quality-$SLURM_JOB_ID
mkdir -p "$output"
common=(--rm --gpus "device=${SLURM_STEP_GPUS}" --cpus=8 --ulimit core=0:0
    --user "$(id -u):$(id -g)" -v "$workspace:/work"
    -v /tmp/relh-generals-gpu/runtime:/work/runtime:ro
    -v /tmp/relh-generals-gpu/source-gpu:/work/oldsource:ro -w /work
    -e HOME=/work -e PYTHONPATH=/work/runtime:/work/source-scripted-benchmark:/work/oldsource
    -e JAX_PLATFORMS=cuda,cpu)
for pair in expander_harvester:sentinel sentinel:expander_harvester expander_harvester:expander_harvester; do
    teacher=${pair%:*}
    opponent=${pair#*:}
    timeout -k 30s 4m docker run "${common[@]}" \
        --name "relh-classic-scripted-$SLURM_JOB_ID-$teacher-$opponent" \
        relh-generals-b300:20260923b \
        python -m integrations.benchmark_coworld_scripted \
        --teacher "$teacher" --opponent "$opponent" --games 256 --seeds 1201 \
        > "$output/$teacher-vs-$opponent.log" 2>&1
    tail -2 "$output/$teacher-vs-$opponent.log"
done
'

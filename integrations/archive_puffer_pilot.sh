#!/usr/bin/env bash
# Preserve the utilization audit while the short evaluation still owns a GPU.
set -euo pipefail

job_id=${1:?pilot Slurm job ID required}
destination=${2:?durable destination required}
workspace=${3:-/tmp/relh-generals-gpu}
[[ "$job_id" =~ ^[0-9]+$ ]]
mkdir -p "$destination"
source_file="$workspace/parallel-eval-gpu-${job_id}.csv"
archive_file="$destination/parallel-eval-gpu-${job_id}.csv"
gpu_step=(srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=1 --chdir=/tmp)

while true; do
    state=$(scontrol show job -o "$job_id" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
    case "$state" in
        PENDING) sleep 30; continue ;;
        RUNNING|COMPLETING) ;;
        *) printf 'Pilot job %s ended: %s\n' "$job_id" "$state"; break ;;
    esac

    temporary=$(mktemp "$destination/.pilot-gpu.XXXXXXXX")
    if "${gpu_step[@]}" cat "$source_file" > "$temporary" 2>/dev/null && [[ -s "$temporary" ]]; then
        source_hash=$("${gpu_step[@]}" sha256sum "$source_file" 2>/dev/null | awk '{print $1}' || true)
        archive_hash=$(sha256sum "$temporary" | awk '{print $1}')
        if [[ -n "$source_hash" && "$source_hash" == "$archive_hash" ]]; then
            if [[ ! -s "$archive_file" ]] || ! cmp -s "$temporary" "$archive_file"; then
                mv "$temporary" "$archive_file"
                printf 'Archived pilot GPU samples: %s\n' "$archive_hash"
            fi
        fi
    fi
    rm -f "$temporary"
    sleep 2
done

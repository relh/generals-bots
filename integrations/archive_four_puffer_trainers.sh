#!/usr/bin/env bash
# Run on the mettabox to preserve four node-local trainer checkpoints.
set -euo pipefail

job_id=${1:?Slurm job ID required}
archive_directory=${2:?durable archive directory required}
workspace=${3:-/tmp/relh-generals-gpu}
[[ "$job_id" =~ ^[0-9]+$ ]]
mkdir -p "$archive_directory"

while true; do
    state=$(scontrol show job -o "$job_id" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
    case "$state" in
        PENDING) sleep 30 ;;
        RUNNING) break ;;
        *) printf 'Job %s ended before archiving began: %s\n' "$job_id" "$state" >&2; exit 1 ;;
    esac
done

pids=()
for index in 0 1 2 3; do
    run_name="gpu-batch-four-long-${job_id}-${index}"
    bash "$(dirname "$0")/archive_puffer_checkpoints.sh" \
        "$job_id" "$workspace/$run_name" "$archive_directory" 60 \
        > "$archive_directory/$run_name-archive.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

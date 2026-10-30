#!/usr/bin/env bash
# Preserve checkpoints from a running GPU allocation before its node-local workspace disappears.
set -euo pipefail

job_id=${1:?Slurm job ID required}
run_directory=${2:?node-local run directory required}
archive_directory=${3:?durable archive directory required}
poll_seconds=${4:-300}
[[ "$job_id" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[0-9]+$ ]]

run_name=$(basename "$run_directory")
checkpoint_directory="checkpoints/metta_generals/$run_name"
mkdir -p "$archive_directory"

gpu_step=(srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=1)

while scontrol show job -o "$job_id" 2>/dev/null | grep -Eq 'JobState=(RUNNING|COMPLETING)'; do
    mapfile -t checkpoints < <(
        "${gpu_step[@]}" find "$run_directory/$checkpoint_directory" -type f -name '*.bin' -printf '%f\n' 2>/dev/null | sort
    )
    for checkpoint in "${checkpoints[@]}"; do
        number=${checkpoint%.bin}
        step=$((10#$number))
        archive="$archive_directory/$run_name-step-$step.tar.gz"
        [[ -s "$archive" ]] && continue

        temporary=$(mktemp "$archive_directory/.checkpoint.XXXXXXXX")
        if "${gpu_step[@]}" tar -C "$run_directory" -czf - run.json "$checkpoint_directory/$checkpoint" > "$temporary"; then
            source_hash=$("${gpu_step[@]}" sha256sum "$run_directory/$checkpoint_directory/$checkpoint" | awk '{print $1}')
            archive_hash=$(tar -xOzf "$temporary" "$checkpoint_directory/$checkpoint" | sha256sum | awk '{print $1}')
            if [[ -n "$source_hash" && "$source_hash" == "$archive_hash" ]]; then
                mv "$temporary" "$archive"
                printf 'Archived step %s: %s %s\n' "$step" "$source_hash" "$archive"
                continue
            fi
            printf 'Checkpoint hash mismatch at step %s\n' "$step" >&2
        fi
        rm -f "$temporary"
        exit 1
    done
    (( poll_seconds > 0 )) || break
    sleep "$poll_seconds"
done

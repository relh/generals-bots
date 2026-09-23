#!/usr/bin/env bash
# Run on the mettabox to preserve node-local evaluation evidence as it appears.
set -euo pipefail

job_id=${1:?evaluation Slurm job ID required}
archive_directory=${2:?durable archive directory required}
workspace=${3:-/tmp/relh-generals-gpu}
pilot_job_id=${4:-}
train_job_id=${5:-}
[[ "$job_id" =~ ^[0-9]+$ ]]
[[ -z "$pilot_job_id" || "$pilot_job_id" =~ ^[0-9]+$ ]]
[[ -z "$train_job_id" || "$train_job_id" =~ ^[0-9]+$ ]]
mkdir -p "$archive_directory"
gpu_step=(srun --jobid="$job_id" --overlap --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=1 --chdir=/tmp)

while true; do
    state=$(scontrol show job -o "$job_id" 2>/dev/null | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
    case "$state" in
        PENDING) sleep 30; continue ;;
        RUNNING|COMPLETING) ;;
        *) printf 'Evaluation job %s ended: %s\n' "$job_id" "$state"; break ;;
    esac

    mapfile -t files < <(
        "${gpu_step[@]}" find "$workspace" -maxdepth 2 -type f \
            \( -name evaluation.json -o -name '*classic*-summary.json' -o -name '*classic*-proof.json' -o -name build.json \
               -o -name "parallel-eval-gpu-${pilot_job_id}.csv" \
               -o -name 'gpu-batch-four-long-*-classic-*-gpu-*.csv' \
               -o -name "gpu-batch-four-long-${train_job_id}-validation-summary.json" \) \
            -print 2>/dev/null | sort
    )
    for file in "${files[@]}"; do
        relative=${file#"$workspace"/}
        case "$relative" in
            gpu-batch-four-long-*-classic-*/*|gpu-batch-four-long-*-classic-*-summary.json|gpu-batch-four-long-*-classic-proof.json|build-classic10-"$job_id"/build.json) ;;
            gpu-batch-four-long-*-classic-*-gpu-*.csv) ;;
            parallel-eval-gpu-"$pilot_job_id".csv|gpu-batch-four-long-"$train_job_id"-validation-summary.json) ;;
            *) continue ;;
        esac
        destination="$archive_directory/${relative//\//__}"
        [[ -s "$destination" ]] && continue
        temporary=$(mktemp "$archive_directory/.evaluation.XXXXXXXX")
        if "${gpu_step[@]}" cat "$file" > "$temporary" \
            && [[ -s "$temporary" ]] \
            && { [[ "$file" != *.json ]] || python3 -m json.tool "$temporary" >/dev/null 2>&1; }; then
            source_hash=$("${gpu_step[@]}" sha256sum "$file" | awk '{print $1}')
            archive_hash=$(sha256sum "$temporary" | awk '{print $1}')
            if [[ "$source_hash" == "$archive_hash" ]]; then
                mv "$temporary" "$destination"
                printf 'Archived %s: %s\n' "$relative" "$source_hash"
                continue
            fi
        fi
        rm -f "$temporary"
    done
    sleep 5
done

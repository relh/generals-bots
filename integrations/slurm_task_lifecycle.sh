#!/usr/bin/env bash
# Source from a host batch script. Callbacks must operate only on this job.
# Required callbacks: task_step_stopped, task_archive_upload, task_cleanup.
# task_step_stopped must confirm remote step completion, not just local PID exit.
task_step_pid=
task_stop_checks=${task_stop_checks:-600}

task_check_space() {
    local path=$1 min_kib=$2 min_inodes=$3 free_kib free_inodes
    free_kib=$(df -Pk "$path" | awk 'NR==2 {print $4}') || return
    free_inodes=$(df -Pi "$path" | awk 'NR==2 {print $4}') || return
    [[ "$free_kib" =~ ^[0-9]+$ && "$free_inodes" =~ ^[0-9]+$ ]] || return 1
    (( free_kib >= min_kib && free_inodes >= min_inodes ))
}

task_run_step() {
    "$@" &
    task_step_pid=$!
    local status=0
    wait "$task_step_pid" || status=$?
    task_step_pid=
    return "$status"
}

task_finish() {
    local status=$? checks=0
    trap - EXIT
    trap '' USR1 TERM INT
    if [[ -n "$task_step_pid" ]]; then
        # srun forwards TERM to the owned step; retain scratch while it exits.
        kill -TERM "$task_step_pid" 2>/dev/null || true
        while kill -0 "$task_step_pid" 2>/dev/null; do
            if (( checks >= task_stop_checks )); then
                printf '%s\n' 'Owned step did not stop; retaining scratch without archiving.' >&2
                exit 125
            fi
            sleep 0.1
            checks=$((checks + 1))
        done
        wait "$task_step_pid" 2>/dev/null || true
        task_step_pid=
    fi
    # The submitter must supply a bounded controller/step completion check.
    # A dead local srun alone does not establish that remote writes stopped.
    if ! task_step_stopped; then
        printf '%s\n' 'Step completion unconfirmed; retaining scratch without archiving.' >&2
        exit 125
    fi
    if ! task_archive_upload; then
        printf '%s\n' 'Result upload failed; retaining scratch and container.' >&2
        exit 74
    fi
    # Failed/signaled runs retain their local evidence even after upload.
    if (( status == 0 )); then
        task_cleanup || exit 74
    fi
    exit "$status"
}

task_install_traps() {
    declare -F task_step_stopped task_archive_upload task_cleanup >/dev/null || return 1
    trap task_finish EXIT
    trap 'exit 124' USR1
    trap 'exit 143' TERM
    trap 'exit 130' INT
}

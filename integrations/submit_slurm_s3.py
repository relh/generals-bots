"""Render or submit one audited job; keep bearer URLs out of public receipts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from integrations.slurm_s3_job import NICE


def render(config, *, name, partition, minutes, cpus, memory_gib):
    if "mount_recovery" in config:
        raise ValueError("Unsupported launch configuration: mount_recovery")
    if not re.fullmatch(r"relh-generals-[a-z0-9-]+", name):
        raise ValueError("Expected a unique relh-generals task job name")
    if partition not in ("rtx4090", "b200", "b300"):
        raise ValueError("Partition is not allowed")
    if partition != "rtx4090" and config.get("required_gpu_memory_gib", 0) <= 24:
        raise ValueError("This single-GPU job needs a measured >24 GiB requirement for a large partition")
    if not 1 <= cpus <= 24 or not 1 <= memory_gib <= 128 or not 11 <= minutes <= 120:
        raise ValueError("Resources exceed this bounded task launcher's limits")
    if config["runtime_seconds"] != minutes * 60:
        raise ValueError("Slurm and credential runtime budgets differ")
    if sum(s["seconds"] for s in config["steps"]) + 600 >= minutes * 60:
        raise ValueError("Step budgets leave insufficient upload/termination time")
    if not config["steps"] or config["steps"][0]["name"] != "smoke":
        raise ValueError("The first step must smoke-test the image")
    storage = config.get("enroot_storage_paths")
    if not isinstance(storage, list) or not storage or any(
            not isinstance(path, str) or not Path(path).is_absolute() for path in storage):
        raise ValueError("Record actual site Enroot unpack/temp filesystems before submission")
    if time.time() + minutes * 60 + 600 >= config["credential_expiry"]:
        raise ValueError("Signing credentials expire too soon")
    code = Path(__file__).with_name("slurm_s3_job.py").read_text()
    # Keep the project-prohibited host out of the shared 4090 partition while
    # leaving Slurm free to choose among the other eligible nodes.
    excluded = "#SBATCH --exclude=metta4\n" if partition == "rtx4090" else ""
    return f'''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
{excluded}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory_gib}G
#SBATCH --time={minutes // 60:02d}:{minutes % 60:02d}:00
#SBATCH --nice={NICE}
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@600
#SBATCH --chdir=/tmp
#SBATCH --output=/tmp/%x-%j.log
set -euo pipefail
umask 077
ulimit -c 0
exec python3 - <<'GENERALS_HOST_RUNNER'
{code}
raise SystemExit(SlurmJob(json.loads({json.dumps(json.dumps(config))})).execute())
GENERALS_HOST_RUNNER
'''


def remote(host, args, **kwargs):
    if host not in ("metta0", "metta1", "metta4"):
        raise ValueError("Use an allowed submit host without changing user identity")
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, shlex.join(args)],
                          capture_output=True, text=True, **kwargs)


def submit(script, receipt_path, public_receipt, host):
    queue = remote(host, ["squeue", "--noheader", "--format=%i|%j"], timeout=30)
    if queue.returncode:
        raise RuntimeError("Cannot reconcile the live task queue")
    for line in queue.stdout.splitlines():
        if re.search(r"(?:relh|richard).*(?:classic|generals)", line, re.I):
            raise RuntimeError("A live Generals task job already exists; reconcile it before submission")
    # Durable intent before contacting sbatch. A lost SSH response must never
    # trigger an automatic duplicate submission; retain intent for reconciliation.
    with receipt_path.open("x") as output:
        json.dump(dict(public_receipt, state="SUBMISSION_INTENT"), output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    result = remote(host, ["sbatch", "--parsable", f"--nice={NICE}"], input=script, timeout=45)
    if result.returncode or not re.fullmatch(r"\d+(?:;[\w.-]+)?\s*", result.stdout):
        raise RuntimeError("Submission response uncertain; reconcile saved intent and queue before any retry")
    job_id = result.stdout.strip().split(";")[0]
    receipt = dict(public_receipt, job_id=job_id, state="SUBMITTED")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    record = remote(host, ["scontrol", "show", "job", "-o", job_id], timeout=30)
    if record.returncode:
        raise RuntimeError("Job ID saved but controller readback failed; do not resubmit")
    # Deliberately select fields: raw controller output may include credentials.
    fields = dict(v.split("=", 1) for v in record.stdout.split() if "=" in v)
    receipt.update({k: fields.get(k) for k in (
        "JobState", "ExitCode", "Nice", "Priority", "Partition", "NodeList", "BatchHost",
        "TimeLimit", "NumCPUs", "ReqTRES", "StartTime", "EndTime")})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if receipt["Nice"] != str(NICE):
        raise RuntimeError("Submitted job Nice differs; saved receipt requires in-place reconciliation")
    return job_id


def monitor(host, job_id, receipt_path):
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
                "BOOT_FAIL", "DEADLINE", "PREEMPTED"}
    previous = None
    while True:
        result = remote(host, ["scontrol", "show", "job", "-o", job_id], timeout=30)
        receipt = json.loads(receipt_path.read_text())
        if result.returncode:
            receipt["readback_error"] = "Controller readback unavailable; reconcile before retry"
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            raise RuntimeError(receipt["readback_error"])
        fields = dict(v.split("=", 1) for v in result.stdout.split() if "=" in v)
        receipt.update({k: fields.get(k) for k in (
            "JobState", "ExitCode", "Nice", "Priority", "Partition", "NodeList", "BatchHost",
            "TimeLimit", "NumCPUs", "ReqTRES", "StartTime", "EndTime", "Reason")})
        receipt["last_controller_readback_epoch"] = time.time()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        state = receipt["JobState"]
        if state != previous:
            print(f"Job {job_id}: {state}, Nice={receipt['Nice']}, Priority={receipt['Priority']}", flush=True)
            previous = state
        if state in terminal:
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Private signed job configuration")
    parser.add_argument("--name", required=True)
    parser.add_argument("--partition", choices=("rtx4090", "b200", "b300"), default="rtx4090")
    parser.add_argument("--minutes", type=int, default=55)
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory-gib", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True, help="Private rendered batch script")
    parser.add_argument("--submit-host", choices=("metta0", "metta1", "metta4"))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    script = render(config, name=args.name, partition=args.partition, minutes=args.minutes,
                    cpus=args.cpus, memory_gib=args.memory_gib)
    with os.fdopen(os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as output:
        output.write(script)
    if args.submit_host:
        if args.receipt is None:
            raise ValueError("A durable receipt path is required before submission")
        job_id = submit(script, args.receipt, dict(config["receipt"], name=args.name), args.submit_host)
        print(job_id, flush=True)
        monitor(args.submit_host, job_id, args.receipt)
    else:
        print("Rendered private batch script; no submission")


if __name__ == "__main__":
    main()

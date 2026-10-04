"""Archive an inactive owned Slurm run without restarting its workload or removing files."""

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from integrations.slurm_s3_job import NICE, SlurmJob


def recover(config):
    original = config["original_receipt"]
    job_id = original["job_id"]
    maintenance_id = os.environ["SLURM_JOB_ID"]
    if (not re.fullmatch(r"\d+", job_id) or not re.fullmatch(r"\d+", maintenance_id)
            or maintenance_id == job_id or original["Nice"] != str(NICE)
            or original["Priority"] != "1"
            or original["JobState"] not in {"FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
            or not re.fullmatch(r"\d+:\d+", original["ExitCode"])):
        raise ValueError("Recovery requires the captured terminal receipt of a different owned job")
    root = Path(config["root"])
    if (root.name != f"relh-generals-{job_id}" or not root.is_absolute()
            or root.is_symlink() or root.resolve() != root
            or root.stat().st_uid != os.getuid() or os.getuid() != config["expected_uid"]
            or not (root / "out").is_dir() or (root / "out").is_symlink()):
        raise ValueError("Retained results path is not owned by the current job identity")
    for extra in ([], ["--steps"]):
        result = subprocess.run(["squeue", *extra, "--noheader", "--format=%i"],
                                capture_output=True, text=True, timeout=15)
        if result.returncode or any(line.strip().split(".")[0] == job_id for line in result.stdout.splitlines()):
            raise RuntimeError("Original job or step is still active, or controller status is unavailable")
    # Never overwrite an earlier recovery attempt or an already published run.
    for name in ("results.tar.gz", "result-manifest.json", "out/receipt.json", "out/batch.log"):
        if (root / name).exists():
            raise FileExistsError("Recovery artifacts already exist; reconcile them before retrying")
    log = Path(config["batch_log"])
    if log.is_symlink() or log.stat().st_uid != os.getuid() or log.parent != root.parent:
        raise ValueError("Original batch log does not belong to the retained run")
    if not re.fullmatch(r"relh-generals-[a-z0-9-]+-" + job_id + r"\.log", log.name):
        raise ValueError("Original batch log name differs from the retained job")
    shutil.copyfile(log, root / "out/batch.log")
    job = SlurmJob.__new__(SlurmJob)
    job.job_id, job.root = job_id, root
    job.receipt = dict(original, recovery_job_id=maintenance_id,
                       recovery_result_prefix=config["result_prefix"],
                       recovery_scope="Archive inactive retained results only; no workload restart or cleanup")
    job.config = {key: config[key] for key in ("output_urls", "manifest_url")}
    job.finalization_deadline = time.monotonic() + 240
    exit_code, exit_signal = map(int, original["ExitCode"].split(":"))
    job.archive_upload(exit_code or (128 + exit_signal if exit_signal else 125))
    print("RETAINED_RESULTS_PUBLISHED original_job=" + job_id + " recovery_job=" + maintenance_id, flush=True)

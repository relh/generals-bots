"""Read-only collection of the current Slurm/S3 result format."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile

from integrations.slurm_s3_job import MAX_PART_BYTES, NICE, extract_input
from integrations.submit_slurm_s3 import remote

TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}
MAX_UNPACKED_BYTES = 32 * 1024**3


def verify_manifest(manifest, expected, job_id):
    receipt, parts = manifest["receipt"], manifest["parts"]
    for key in ("source_revision", "input_sha256", "image_sha256", "result_prefix"):
        if receipt[key] != expected[key]:
            raise ValueError("Result source or artifact identity differs: " + key)
    if receipt["job_id"] != job_id or receipt["Nice"] != str(NICE) or receipt["Priority"] != "1":
        raise ValueError("Result job or scheduling identity differs")
    if not 1 <= len(parts) <= 4 or [p["index"] for p in parts] != list(range(len(parts))):
        raise ValueError("Result part indexes or count differ")
    for part in parts:
        if (isinstance(part["bytes"], bool) or not isinstance(part["bytes"], int)
                or not 0 < part["bytes"] <= MAX_PART_BYTES
                or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])):
            raise ValueError("Result part size or digest is invalid")
    return sum(part["bytes"] for part in parts)


def sandbox_session(profile):
    """Refresh the configured SSO role in memory without changing login files."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from datetime import datetime
    import time

    prior = boto3.Session(profile_name=profile)
    scoped = prior._session.get_scoped_config()
    name = scoped.get("sso_session")
    settings = prior._session.full_config["sso_sessions"][name] if name else scoped
    cache_key = hashlib.sha1((name or settings["sso_start_url"]).encode()).hexdigest()
    cached = json.loads((Path.home() / ".aws/sso/cache" / (cache_key + ".json")).read_text())
    if datetime.fromisoformat(cached["expiresAt"].replace("Z", "+00:00")).timestamp() <= time.time() + 600:
        raise RuntimeError("Existing SSO login expires too soon for result collection")
    role = boto3.client("sso", region_name=settings["sso_region"], config=Config(signature_version=UNSIGNED)).get_role_credentials(
        roleName=scoped["sso_role_name"], accountId=scoped["sso_account_id"], accessToken=cached["accessToken"])["roleCredentials"]
    fresh = boto3.Session(aws_access_key_id=role["accessKeyId"], aws_secret_access_key=role["secretAccessKey"],
                          aws_session_token=role["sessionToken"], region_name="us-east-1")
    before, after = (s.client("sts", region_name="us-east-1").get_caller_identity() for s in (prior, fresh))
    if before["Account"] != after["Account"] or before["Arn"].split("/")[1] != after["Arn"].split("/")[1]:
        raise RuntimeError("Result collection role differs from the configured sandbox role")
    return fresh


def collect(launch, submission, output, *, host="metta0", profile="sandbox", controller_expired=False):
    job_id = str(submission["job_id"])
    if not re.fullmatch(r"\d+", job_id):
        raise ValueError("Expected a numeric controller job ID")
    result = remote(host, ["scontrol", "show", "job", "-o", job_id], timeout=30)
    expired = bool(result.returncode)
    if expired:
        if not controller_expired or "Invalid job id specified" not in result.stderr:
            raise RuntimeError("Controller readback failed")
        if submission.get("Nice") != str(NICE) or submission.get("Priority") != "1":
            raise ValueError("Expired controller requires authentic saved submission scheduling identity")
        active = remote(host, ["squeue", "--jobs=" + job_id, "--noheader", "--format=%i"], timeout=30)
        if (active.returncode and "Invalid job id specified" not in active.stderr) or active.stdout.strip():
            raise RuntimeError("Expired controller job must be absent from the active queue")
        controller = {"available": False, "reason": "Controller retention expired", "JobId": job_id}
    else:
        fields = dict(item.split("=", 1) for item in result.stdout.split() if "=" in item)
        controller = {key: fields.get(key) for key in ("JobId", "JobState", "ExitCode", "Nice", "Priority", "NodeList", "TimeLimit", "RunTime")}
        if controller["JobId"] != job_id or controller["Nice"] != str(NICE) or controller["Priority"] != "1":
            raise ValueError("Controller job identity or scheduling policy differs")
        if controller["JobState"] not in TERMINAL:
            return dict(job_id=job_id, state=controller["JobState"], collected=False)
    from botocore.config import Config

    client = sandbox_session(profile).client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
    expected = launch["public"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "controller.json").write_text(json.dumps(controller, indent=2) + "\n")
    def download(key, path):
        try:
            client.download_file(expected["bucket"], key, str(path))
        except Exception:
            raise RuntimeError("S3 result read failed; local evidence remains preserved") from None
    prefix = expected["result_prefix"]
    download(prefix + "/manifest.json", output / "manifest.json")
    manifest = json.loads((output / "manifest.json").read_text())
    compressed = verify_manifest(manifest, expected, job_id)
    if expired:
        marker = manifest["receipt"]
        if type(marker.get("workload_exit_code")) is not int or marker["workload_exit_code"] < 0:
            raise ValueError("Expired controller requires the signed runner finalization marker")
        (output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    reserve = 512 * 1024**2
    if shutil.disk_usage(output).free < 2 * compressed + reserve:
        raise RuntimeError("Insufficient collection space; manifest preserved and remote results untouched")
    archive = output / "results.tar.gz"
    with archive.open("xb") as joined:
        for part in manifest["parts"]:
            path = output / f"result.part{part['index']:03d}"
            download(prefix + "/" + path.name, path)
            with path.open("rb") as stream:
                if path.stat().st_size != part["bytes"] or hashlib.file_digest(stream, "sha256").hexdigest() != part["sha256"]:
                    raise ValueError("Result part differs from its manifest; received bytes remain preserved")
                stream.seek(0)
                shutil.copyfileobj(stream, joined)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        unpacked = sum(member.size for member in members)
        if any(member.name != "out" and not member.name.startswith("out/") for member in members):
            raise ValueError("Result members must remain under out/")
        if unpacked > MAX_UNPACKED_BYTES or len(members) > 100000:
            raise ValueError("Result extraction exceeds its bounded byte or member budget")
    stat = os.statvfs(output)
    if stat.f_bavail * stat.f_frsize < unpacked + reserve or stat.f_favail < len(members) + 1000:
        raise RuntimeError("Insufficient extraction space or inodes; verified archive and parts preserved")
    extract_input(archive, output, MAX_UNPACKED_BYTES, 100000)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    proof = dict(job_id=job_id, controller=controller, source_revision=expected["source_revision"],
                 compressed_bytes=compressed, unpacked_bytes=unpacked, members=len(members), archive_sha256=digest,
                 parts_verified=True, safe_extraction=True, archive_and_parts_preserved=True, collected=True,
                 controller_expired=expired, workload_exit_code=manifest["receipt"].get("workload_exit_code"))
    (output / "collection-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True, help="Sanitized public launch manifest")
    parser.add_argument("--receipt", type=Path, required=True, help="Saved submission receipt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="metta0")
    parser.add_argument("--profile", default="sandbox")
    parser.add_argument("--controller-expired", action="store_true", help="Require final signed S3 marker and authentic submission when controller retention expired")
    args = parser.parse_args()
    try:
        result = collect(json.loads(args.launch.read_text()), json.loads(args.receipt.read_text()), args.output,
                         host=args.host, profile=args.profile, controller_expired=args.controller_expired)
    except Exception as error:
        # Transport/login errors may carry private bearer data. Only our own
        # bounded messages can be shown to the caller.
        if isinstance(error, (RuntimeError, ValueError)):
            raise SystemExit(str(error)) from None
        raise SystemExit("Result collection failed: " + type(error).__name__) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Host-side, single-job Pyxis execution with bounded S3 transfers.

The submitter embeds this module in a host sbatch script. Credentials remain
in memory; only the public receipt is written into the result directory.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import time


MAX_PART_BYTES = 3_500_000_000
NICE = 2147483645


class JobSignal(Exception):
    def __init__(self, signum):
        self.status = 128 + signum
        if signum == signal.SIGUSR1:
            self.status = 124


def check_space(path, required_bytes, required_inodes):
    stat = os.statvfs(path)
    if stat.f_bavail * stat.f_frsize < required_bytes or stat.f_favail < required_inodes:
        raise RuntimeError("Insufficient free bytes or inodes on owned scratch filesystem")


def extract_input(archive, destination, max_bytes, max_members):
    # Validate the entire member list before any extraction. Inputs contain no
    # links, devices, environments, or paths escaping the owned directory.
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > max_members or sum(m.size for m in members) > max_bytes:
            raise ValueError("Input archive exceeds its declared unpacking budget")
        names = set()
        for member in members:
            path = Path(member.name)
            if (path.is_absolute() or ".." in path.parts or member.name in names
                    or not (member.isfile() or member.isdir())):
                raise ValueError("Unsafe input archive member")
            names.add(member.name)
        source.extractall(destination, members=members)


def transfer(url, path, upload=False, deadline=None):
    if not url.startswith("https://") or any(c in url for c in '\r\n"\\'):
        raise ValueError("Invalid presigned URL")
    # Config arrives on stdin, never in argv, output logs or exceptions.
    seconds = min(180, int(deadline - time.monotonic() - 10)) if deadline is not None else 180
    if seconds < 5:
        raise RuntimeError("Transfer deadline exhausted; retain local artifacts")
    command = ["curl", "--silent", "--fail", "--connect-timeout", "15",
               "--max-time", str(seconds),
               "--config", "-"]
    command += ["--upload-file" if upload else "--output", str(path)]
    result = subprocess.run(command, input=f'url = "{url}"\n', text=True,
                            capture_output=True, timeout=seconds + 5)
    if result.returncode:
        raise RuntimeError(f"S3 {'upload' if upload else 'download'} failed (curl {result.returncode})")


class SlurmJob:
    def __init__(self, config):
        self.config = config
        self.job_id = os.environ["SLURM_JOB_ID"]
        if not re.fullmatch(r"\d+", self.job_id):
            raise ValueError("Invalid job ID")
        self.root = Path(config["scratch_parent"]) / f"relh-generals-{self.job_id}"
        self.container = f"relh-generals-{self.job_id}"
        self.process = None
        self.step_started = False
        self.owns_root = False
        self.runtime_env = dict(os.environ)
        self.receipt = dict(config["receipt"], job_id=self.job_id,
                            node=os.environ.get("SLURMD_NODENAME", "unknown"))

    def check_priority(self):
        record = subprocess.check_output(
            ["scontrol", "show", "job", "-o", self.job_id], text=True, timeout=15)
        fields = dict(item.split("=", 1) for item in record.split() if "=" in item)
        if fields.get("Nice") != str(NICE):
            raise RuntimeError("Controller did not confirm maximum positive Nice")
        if fields.get("TimeLimit") in (None, "UNLIMITED", "NOT_SET"):
            raise RuntimeError("Controller did not confirm finite runtime")
        self.receipt.update({key: fields.get(key) for key in (
            "Nice", "Priority", "TimeLimit", "Partition", "NumCPUs", "ReqTRES", "NodeList")})

    def prepare(self):
        self.check_priority()
        # Refuse reuse, including accidental restarts with the same job ID.
        self.root.mkdir(mode=0o700)
        self.owns_root = True
        (self.root / "out").mkdir()
        check_space(self.root, self.config["scratch_bytes"], self.config["scratch_inodes"])
        # Keep Enroot's unpack/cache trees on the checked, owned filesystem.
        # The shared /tmp may have free bytes but no inodes.
        for variable, relative in (("ENROOT_TEMP_PATH", "enroot-tmp"),
                                   ("ENROOT_CACHE_PATH", "enroot-cache"),
                                   ("ENROOT_DATA_PATH", "enroot-data"),
                                   ("ENROOT_RUNTIME_PATH", "enroot-runtime")):
            path = self.root / relative
            path.mkdir(mode=0o700)
            self.runtime_env[variable] = str(path)
        self.runtime_env["ENROOT_MAX_PROCESSORS"] = str(self.config.get("cpus", 8))
        if time.time() + self.config["runtime_seconds"] + 600 >= self.config["credential_expiry"]:
            raise RuntimeError("Signing credentials expire too soon for this allocation")
        archive = self.root / "input.tar.gz"
        transfer(self.config["input_url"], archive)
        with archive.open("rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != self.config["input_sha256"]:
            raise RuntimeError("Input digest mismatch")
        extract_input(archive, self.root / "input", self.config["input_unpacked_bytes"],
                      self.config["input_members"])
        if self.config.get("image_parts"):
            image = self.root / "input/image.sqsh"
            if image.exists():
                raise ValueError("Image must come from exactly one declared input source")
            digest = hashlib.sha256()
            with image.open("xb") as target:
                for index, part in enumerate(self.config["image_parts"]):
                    path = self.root / f"image-input.part{index:03d}"
                    transfer(part["url"], path)
                    if path.stat().st_size != part["bytes"] or part["bytes"] >= 4_000_000_000:
                        raise ValueError("Image input part size differs")
                    part_digest = hashlib.sha256()
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            part_digest.update(block)
                            digest.update(block)
                            target.write(block)
                    if part_digest.hexdigest() != part["sha256"]:
                        raise ValueError("Image input part digest differs")
                    path.unlink()  # Verified bytes are retained in the owned image.
            if digest.hexdigest() != self.config["image_sha256"]:
                raise ValueError("Reassembled image digest differs")

    def run_step(self, name, argv, seconds):
        # Repeat immediately before Pyxis/Enroot may unpack an image.
        check_space(self.root, self.config["image_unpacked_bytes"], self.config["image_inodes"])
        check_space(self.runtime_env["ENROOT_TEMP_PATH"], self.config["image_unpacked_bytes"],
                    self.config["image_inodes"])
        image = self.config["image"]
        if image == "input/image.sqsh":
            image = str(self.root / image)
        mounts = f"{self.root}:/work"
        if self.config.get("mount_recovery"):
            mounts += f",{self.root}/input/recovery:/recovery:ro"
        command = ["srun", f"--nice={NICE}", "--nodes=1", "--ntasks=1",
                   "--kill-on-bad-exit=1", "--unbuffered",
                   f"--container-image={image}",
                   f"--container-name={self.container}", "--no-container-mount-home",
                   f"--container-mounts={mounts}", "--container-workdir=/work", *argv]
        with (self.root / "out" / f"{name}.log").open("wb") as log:
            self.step_started = True
            # A signal between spawn and PID assignment must not orphan a step
            # that finalization cannot wait for. Restore the mask in the child.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1, signal.SIGTERM, signal.SIGINT})
            try:
                self.process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=self.runtime_env,
                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, mask))
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            try:
                code = self.process.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{name} exceeded its bounded runtime") from None
            # Keep the Popen reference until finalization confirms remote steps.
            if code:
                raise subprocess.CalledProcessError(code, [name])
        if not self.steps_stopped():
            raise RuntimeError("Remote step completion not confirmed")
        self.process = None

    def steps_stopped(self):
        if not self.step_started:
            return True
        result = subprocess.run(["squeue", "--steps", f"--jobs={self.job_id}",
                                 "--noheader", "--format=%i"], capture_output=True,
                                text=True, timeout=15)
        if result.returncode:
            return False
        # batch/extern last until job exit; numbered steps must be gone.
        return all(line.strip() in (f"{self.job_id}.batch", f"{self.job_id}.extern")
                   for line in result.stdout.splitlines() if line.strip())

    def stop_and_wait(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()  # srun forwards TERM to this job's step.
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if self.steps_stopped():
                return True
            time.sleep(1)
        return False

    def archive_upload(self, status):
        self.receipt["workload_exit_code"] = status
        name = os.environ.get("SLURM_JOB_NAME", "")
        if re.fullmatch(r"relh-generals-[a-z0-9-]+", name):
            batch_log = Path("/tmp") / f"{name}-{self.job_id}.log"
            if batch_log.is_file():
                shutil.copyfile(batch_log, self.root / "out/batch.log")
        (self.root / "out" / "receipt.json").write_text(json.dumps(self.receipt, indent=2) + "\n")
        archive = self.root / "results.tar.gz"
        seconds = min(120, int(self.finalization_deadline - time.monotonic() - 60))
        if seconds <= 0:
            raise RuntimeError("Insufficient finalization time to archive results")
        subprocess.run(["tar", "-czf", str(archive), "-C", str(self.root), "out"],
                       check=True, timeout=seconds, capture_output=True)
        part_count = (archive.stat().st_size + MAX_PART_BYTES - 1) // MAX_PART_BYTES
        if part_count > len(self.config["output_urls"]):
            raise RuntimeError("Results exceed separately signed output part capacity")
        parts = []
        with archive.open("rb") as source:
            for index in range(part_count):
                part = self.root / f"results.part{index:03d}"
                digest = hashlib.sha256()
                size = 0
                with part.open("wb") as target:
                    while size < MAX_PART_BYTES:
                        if time.monotonic() + 15 >= self.finalization_deadline:
                            raise RuntimeError("Result packing deadline exhausted")
                        block = source.read(min(1024 * 1024, MAX_PART_BYTES - size))
                        if not block:
                            break
                        target.write(block)
                        digest.update(block)
                        size += len(block)
                transfer(self.config["output_urls"][index], part, upload=True,
                         deadline=self.finalization_deadline)
                parts.append(dict(index=index, bytes=size, sha256=digest.hexdigest()))
        manifest = self.root / "result-manifest.json"
        manifest.write_text(json.dumps(dict(receipt=self.receipt, parts=parts), indent=2) + "\n")
        # This final upload is the result-completion marker.
        transfer(self.config["manifest_url"], manifest, upload=True, deadline=self.finalization_deadline)

    def cleanup(self):
        if self.step_started:
            subprocess.run(["enroot", "remove", "--force", f"pyxis_{self.container}"],
                           check=True, timeout=60, capture_output=True, env=self.runtime_env)
        # root was created exclusively by this process and is never reused.
        shutil.rmtree(self.root)

    def execute(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
        try:
            return self._execute()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _execute(self):
        status = 0
        def interrupted(signum, _frame):
            raise JobSignal(signum)
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupted)
        try:
            self.receipt["phase"] = "prepare"
            self.prepare()
            for step in self.config["steps"]:
                self.receipt["phase"] = step["name"]
                self.run_step(step["name"], step["argv"], step["seconds"])
        except JobSignal as error:
            status = error.status
        except subprocess.CalledProcessError as error:
            status = error.returncode if error.returncode > 0 else 128 - error.returncode
        except Exception as error:
            # Never stringify arbitrary transport exceptions: URLs are secrets.
            reason = type(error).__name__
            if type(error) in (RuntimeError, ValueError):
                reason += ": " + str(error)
            self.receipt["failure"] = reason
            print(f"Workload failed in {self.receipt['phase']}: {reason}", flush=True)
            status = 1
        finally:
            for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
        if not self.owns_root or not (self.root / "out").exists():
            return status or 1
        self.finalization_deadline = time.monotonic() + 510
        try:
            if not self.stop_and_wait():
                print("Step completion unconfirmed; retaining scratch without archiving", flush=True)
                return 125
            self.archive_upload(status)
            if status == 0:
                self.cleanup()
        except Exception as error:
            print(f"Finalization failed: {type(error).__name__}; retaining scratch", flush=True)
            return 74
        return status

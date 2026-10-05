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


def gpu_query(*arguments):
    """Only NVIDIA query output is included in errors; never transport credentials."""
    result = subprocess.run(["nvidia-smi", *arguments], capture_output=True,
                            text=True, timeout=15)
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()[:2000]
        raise RuntimeError(f"nvidia-smi query exit {result.returncode}: {detail}")
    return result.stdout.strip()


def allocated_gpu_identity(environ=None):
    """Resolve the single device visible inside Slurm's constrained device cgroup.

    SLURM_JOB_GPUS/STEP_GPUS are global GRES identifiers, not NVML ordinals
    inside that cgroup. Never pass those numeric identifiers to nvidia-smi -i.
    """
    environ = os.environ if environ is None else environ
    assigned = environ.get("SLURM_STEP_GPUS") or environ.get("SLURM_JOB_GPUS", "")
    if not re.fullmatch(r"(?:GPU-[a-fA-F0-9-]+|[0-9]+)", assigned):
        raise RuntimeError("Expected one controller-provided GPU assignment")
    rows = gpu_query("--query-gpu=index,uuid", "--format=csv,noheader,nounits").splitlines()
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one cgroup-visible GPU; observed {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if (len(fields) != 2 or not fields[0].isdigit()
            or not re.fullmatch(r"GPU-[a-fA-F0-9-]+", fields[1])):
        raise RuntimeError("Unrecognized visible GPU identity")
    if assigned.startswith("GPU-") and assigned != fields[1]:
        raise RuntimeError("Visible GPU UUID differs from controller assignment")
    return dict(slurm_assignment=assigned, visible_index=fields[0], uuid=fields[1])


def verify_allocated_gpu_idle():
    identity = allocated_gpu_identity()
    processes = gpu_query("--id=" + identity["uuid"], "--query-compute-apps=pid",
                          "--format=csv,noheader,nounits")
    if processes:
        raise RuntimeError("Allocated physical GPU already has compute processes; leaving them untouched")
    row = gpu_query("--id=" + identity["uuid"],
                    "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits")
    fields = [field.strip() for field in row.split(",")]
    if len(fields) != 3 or fields[0] != identity["uuid"] or float(fields[1]) >= 2048 or float(fields[2]) >= 20:
        raise RuntimeError("Allocated physical GPU is not idle before workload startup")
    identity.update(memory_mib=float(fields[1]), utilization_percent=float(fields[2]))
    return identity


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


def download_parts(parts):
    """Fetch at most four parts with one owned, interruptible curl process."""
    if not 1 <= len(parts) <= 4:
        raise ValueError("Parallel download requires one to four parts")
    sections = []
    for url, path in parts:
        if not url.startswith("https://") or any(c in url for c in '\r\n"\\'):
            raise ValueError("Invalid presigned URL")
        path = str(path)
        if any(c in path for c in '\r\n"\\'):
            raise ValueError("Invalid download path")
        sections.append(f'url = "{url}"\noutput = "{path}"\n'
                        'silent\nfail\nconnect-timeout = 15\nmax-time = 600\n')
    # subprocess.run kills and waits for this sole child on signal exceptions
    # as well as timeouts. No downloader may outlive preparation/finalization.
    result = subprocess.run(
        ["curl", "--parallel", "--parallel-max", "4", "--fail-early", "--config", "-"],
        input="next\n".join(sections), text=True, capture_output=True, timeout=605,
    )
    if result.returncode:
        raise RuntimeError(f"S3 parallel download failed (curl {result.returncode})")


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

    def retire_verified_container(self):
        """Remove one explicitly named old task container, preserving its scratch.

        The caller supplies the SHA of the fully collected, S3-verified archive.
        Neither retained source/results nor any other container is removed.
        """
        retained = self.config.get("retained_container")
        if retained is None:
            return
        prior = retained.get("job_id", "")
        expected = retained.get("results_sha256", "")
        if (not isinstance(prior, str) or not re.fullmatch(r"[1-9][0-9]*", prior)
                or int(prior) >= int(self.job_id)
                or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ValueError("Invalid retained task container identity")
        root = self.root.parent / f"relh-generals-{prior}"
        archive = root / "results.tar.gz"
        receipt = root / "out/receipt.json"
        for path in (root, root / "out", archive, receipt):
            if path.is_symlink() or path.stat().st_uid != os.getuid():
                raise ValueError("Retained task path ownership differs")
        if json.loads(receipt.read_text()).get("job_id") != prior:
            raise ValueError("Retained receipt belongs to another job")
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Retained archive differs from verified collected results")
        for steps in (False, True):
            command = ["squeue", "--noheader", "--format=%i"]
            if steps:
                command.append("--steps")
            state = subprocess.run(command, capture_output=True, text=True, timeout=15)
            if state.returncode:
                raise RuntimeError("Cannot confirm retained job is inactive")
            if any(line.strip().split(".")[0].split("_")[0] == prior
                   for line in state.stdout.splitlines()):
                raise RuntimeError("Retained job or step remains active")
        def enroot(*args):
            result = subprocess.run(["enroot", *args], capture_output=True, text=True,
                                    timeout=60, env=self.runtime_env)
            if result.returncode:
                raise RuntimeError("Retained owned container operation failed")
            return set(result.stdout.splitlines())
        allowed = {f"pyxis_relh-generals-{prior}", f"pyxis_{prior}_relh-generals-{prior}"}
        removed = sorted(enroot("list") & allowed)
        for name in removed:
            enroot("remove", "--force", name)
        if enroot("list") & allowed:
            raise RuntimeError("Retained owned container remains after removal")
        self.receipt["retired_owned_container"] = dict(
            job_id=prior, results_sha256=expected, removed=removed,
            retained_scratch=str(root), source_and_results_preserved=True)

    def prepare(self):
        self.check_priority()
        # Refuse reuse, including accidental restarts with the same job ID.
        self.root.mkdir(mode=0o700)
        self.owns_root = True
        (self.root / "out").mkdir()
        # Pyxis filters these job environment overrides. Host enroot commands
        # must use the same site defaults, or cleanup targets an empty tree.
        for variable in ("ENROOT_TEMP_PATH", "ENROOT_CACHE_PATH", "ENROOT_DATA_PATH",
                         "ENROOT_RUNTIME_PATH", "ENROOT_LIBRARY_PATH", "ENROOT_SYSCONF_PATH"):
            self.runtime_env.pop(variable, None)
        self.runtime_env["ENROOT_MAX_PROCESSORS"] = str(self.config.get("cpus", 8))
        self.retire_verified_container()
        check_space(self.root, self.config["scratch_bytes"], self.config["scratch_inodes"])
        self.receipt["gpu_environment"] = {
            key: os.environ.get(key) for key in
            ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}
        # A batch shell is not the GPU execution context. Record its visibility,
        # but enforce device ownership/occupancy inside the allocated Pyxis step.
        try:
            self.receipt["batch_gpu_visibility"] = gpu_query(
                "--query-gpu=index,uuid", "--format=csv,noheader,nounits")
        except RuntimeError as error:
            self.receipt["batch_gpu_visibility"] = str(error)
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
            parallelism = self.config.get("image_download_parallelism", 1)
            if isinstance(parallelism, bool) or not isinstance(parallelism, int) or not 1 <= parallelism <= 4:
                raise ValueError("Image download parallelism must be one to four")
            started = time.monotonic()
            image = self.root / "input/image.sqsh"
            if image.exists():
                raise ValueError("Image must come from exactly one declared input source")
            digest = hashlib.sha256()
            with image.open("xb") as target:
                for index, part in enumerate(self.config["image_parts"]):
                    path = self.root / f"image-input.part{index:03d}"
                    if parallelism == 1:
                        transfer(part["url"], path)
                    elif index % parallelism == 0:
                        group = self.config["image_parts"][index:index + parallelism]
                        download_parts([(entry["url"], self.root / f"image-input.part{index + offset:03d}")
                                        for offset, entry in enumerate(group)])
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
            self.receipt["image_download"] = dict(
                parallelism=parallelism, seconds=time.monotonic() - started, bytes=image.stat().st_size,
            )


    def run_step(self, name, argv, seconds):
        # Repeat immediately before Pyxis/Enroot may unpack an image.
        check_space(self.root, self.config["image_unpacked_bytes"], self.config["image_inodes"])
        # These are the actual site-configured unpack/temp filesystems, verified
        # before submission. Never assume filtered ENROOT_* overrides applied.
        for path in self.config.get("enroot_storage_paths", []):
            check_space(path, self.config["image_unpacked_bytes"], self.config["image_inodes"])
        image = self.config["image"]
        if image == "input/image.sqsh":
            image = str(self.root / image)
        mounts = f"{self.root}:/work"
        if self.config.get("mount_recovery"):
            mounts += f",{self.root}/input/recovery:/recovery:ro"
        print(f"STEP_START {name} budget_seconds={seconds}", flush=True)
        command = ["srun", f"--nice={NICE}", "--nodes=1", "--ntasks=1", "--gres=gpu:1",
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
            deadline = time.monotonic() + seconds
            previous_progress = None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"{name} exceeded its bounded runtime")
                try:
                    code = self.process.wait(timeout=min(30, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if name == "train":
                        progress = self.training_progress()
                        if progress and progress != previous_progress:
                            print(progress, flush=True)
                            previous_progress = progress
            # Keep the Popen reference until finalization confirms remote steps.
            if code:
                raise subprocess.CalledProcessError(code, [name])
        # A successful srun can precede controller retirement of its step.
        # Do not attach external srun observers: they
        # participate in this barrier and can prevent the next phase starting.
        # Monitor these host-log markers and retained results instead.
        if not self.stop_and_wait():
            raise RuntimeError("Remote step completion not confirmed")
        self.process = None
        print(f"STEP_DONE {name}", flush=True)

    def training_progress(self):
        """Expose numeric Puffer progress from the batch host, without a step.

        Read only a bounded tail of our console. Never forward raw log text:
        the public batch log must not accidentally expose credentials.
        This is observation only; the workload owns its throughput gate.
        """
        path = self.root / "out/run/console.log"
        try:
            with path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(0, source.tell() - 131072))
                history = source.read(131072).decode(errors="replace")
        except OSError:
            return None
        times = {}
        for block in history.split("╭"):
            epoch = re.search(r"Epoch\s+(\d+)", block)
            uptime = re.search(r"Uptime\s+((?:\d+(?:ms|[dhms])\s*)+)", block)
            if epoch and uptime:
                factors = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": .001}
                times[int(epoch[1])] = sum(int(v) * factors[u] for v, u in
                                          re.findall(r"(\d+)(ms|[dhms])", uptime[1]))
        if not times:
            return None
        last = max(times)
        progress = f"TRAIN_PROGRESS epoch={last} uptime_seconds={times[last]:.3f}"
        first = last - 2
        games = self.config.get("receipt", {}).get("environment_count")
        horizon = self.config.get("receipt", {}).get("horizon")
        if (first in times and times[last] > times[first]
                and type(games) is int and games > 0
                and type(horizon) is int and horizon > 0):
            sps = 2 * games * horizon / (times[last] - times[first])
            progress += f" environment_sps_last_two_epochs={sps:.2f}"
        return progress

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
            def enroot(*arguments):
                result = subprocess.run(["enroot", *arguments], timeout=60,
                                        capture_output=True, text=True, env=self.runtime_env)
                if result.returncode:
                    # This command contains only list/remove and our container
                    # name, never transport URLs. Retain its real diagnostic.
                    detail = (result.stdout + result.stderr).strip()[:2000]
                    raise RuntimeError(f"Owned container cleanup exit {result.returncode}: {detail}")
                return set(result.stdout.splitlines())
            allowed = {f"pyxis_{self.container}", f"pyxis_{self.job_id}_{self.container}"}
            owned = enroot("list") & allowed
            for name in sorted(owned):
                enroot("remove", "--force", name)
            if enroot("list") & allowed:
                raise RuntimeError("Owned container remains after cleanup")
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
            # Only our explicit RuntimeErrors are safe to stringify. Transport
            # exceptions can contain credentials and remain type-only.
            detail = ": " + str(error) if type(error) is RuntimeError else ""
            print(f"Finalization failed: {type(error).__name__}{detail}; retaining scratch", flush=True)
            return 74
        return status

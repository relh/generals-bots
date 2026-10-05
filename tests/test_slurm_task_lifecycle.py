"""CPU-only lifecycle tests; these never invoke Slurm or access S3."""
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest


LIBRARY = Path(__file__).resolve().parents[1] / "integrations/slurm_task_lifecycle.sh"


class TaskLifecycleTests(unittest.TestCase):
    def exercise(self, mode, sent_signal=None, upload_fail=False, stopped=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text('''import pathlib, signal, sys, time
p = pathlib.Path("events")
def emit(s):
    with p.open("a") as f: f.write(s + "\\n")
def terminate(*args):
    emit("CHECKPOINT")
    time.sleep(.15)
    emit("LAST_WRITE")
    pathlib.Path("writing").unlink()
    emit("STOPPED")
    sys.exit(0)
signal.signal(signal.SIGTERM, terminate)
pathlib.Path("writing").touch()
emit("READY")
if sys.argv[1] == "wait":
    while True: time.sleep(.02)
pathlib.Path("writing").unlink()
emit("STOPPED")
sys.exit(7 if sys.argv[1] == "fail" else 0)
''')
            script = f'''set -euo pipefail
source {shlex.quote(str(LIBRARY))}
task_step_stopped() {{ test ! -e writing && {"true" if stopped else "false"}; }}
task_archive_upload() {{
    test ! -e writing || {{ echo STEP_STILL_WRITING >>events; return 1; }}
    echo UPLOAD >>events
    {"return 1" if upload_fail else "return 0"}
}}
task_cleanup() {{ echo CLEANUP >>events; }}
task_install_traps
task_run_step {shlex.quote(sys.executable)} worker.py {shlex.quote(mode)}
'''
            proc = subprocess.Popen(["bash", "-c", script], cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True)
            try:
                if sent_signal:
                    deadline = time.monotonic() + 5
                    while not (root / "events").exists() or "READY" not in (root / "events").read_text():
                        if time.monotonic() > deadline:
                            self.fail("Worker startup timed out")
                        time.sleep(.01)
                    proc.send_signal(sent_signal)
                _, stderr = proc.communicate(timeout=10)
                events = (root / "events").read_text().splitlines()
                self.assertNotIn("STEP_STILL_WRITING", events)
                if "UPLOAD" in events:
                    self.assertLess(events.index("STOPPED"), events.index("UPLOAD"))
                return proc.returncode, events, stderr.decode()
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate()

    def test_success(self):
        code, events, _ = self.exercise("success")
        self.assertEqual(code, 0)
        self.assertEqual(events[-2:], ["UPLOAD", "CLEANUP"])

    def test_step_failure_preserves_status_and_scratch(self):
        code, events, _ = self.exercise("fail")
        self.assertEqual(code, 7)
        self.assertEqual(events[-1], "UPLOAD")
        self.assertNotIn("CLEANUP", events)

    def test_signals_wait_for_checkpoint_and_last_write(self):
        for sig, expected in [(signal.SIGUSR1, 124), (signal.SIGTERM, 143), (signal.SIGINT, 130)]:
            with self.subTest(signal=sig):
                code, events, _ = self.exercise("wait", sent_signal=sig)
                self.assertEqual(code, expected)
                self.assertEqual(events[-4:], ["CHECKPOINT", "LAST_WRITE", "STOPPED", "UPLOAD"])
                self.assertNotIn("CLEANUP", events)

    def test_upload_failure_retains_scratch_and_container(self):
        code, events, stderr = self.exercise("success", upload_fail=True)
        self.assertEqual(code, 74)
        self.assertNotIn("CLEANUP", events)
        self.assertIn("retaining scratch and container", stderr)

    def test_unconfirmed_remote_completion_prevents_archive(self):
        code, events, _ = self.exercise("success", stopped=False)
        self.assertEqual(code, 125)
        self.assertNotIn("UPLOAD", events)
        self.assertNotIn("CLEANUP", events)

    def test_space_and_inode_gates(self):
        for available, inodes, expected in [(100, 100, 0), (1, 100, 1), (100, 1, 1)]:
            script = f'''source {shlex.quote(str(LIBRARY))}
df() {{
  echo 'Filesystem blocks used available capacity mount'
  if [[ "$1" == -Pk ]]; then echo 'mock 100 0 {available} 0 /owned';
  else echo 'mock 100 0 {inodes} 0 /owned'; fi
}}
task_check_space /owned 50 50
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True)
            self.assertEqual(result.returncode, expected)


if __name__ == "__main__":
    unittest.main()

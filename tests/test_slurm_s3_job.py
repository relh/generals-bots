"""CPU integration harness: mock only Slurm/Pyxis/S3, run real child processes."""
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from integrations.slurm_s3_job import extract_input, SlurmJob


class S3JobTests(unittest.TestCase):
    def run_job(self, mode="success", sent_signal=None, fail_upload=False, low_nice=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / "bin"
            commands.mkdir()
            archive = root / "input.tar.gz"
            with tarfile.open(archive, "w:gz") as target:
                member = tarfile.TarInfo("source.txt")
                member.size = 4
                target.addfile(member, io.BytesIO(b"data"))
            scripts = {
                "scontrol": f"print('JobId=999 Nice={100 if low_nice else 2147483645} Priority=1 TimeLimit=00:10:00')",
                "squeue": "import pathlib; print('999.0' if pathlib.Path('writing').exists() else '')",
                "enroot": "import pathlib; assert not pathlib.Path('writing').exists(); pathlib.Path('cleaned').touch()",
                "srun": '''import os,sys
assert '--nice=2147483645' in sys.argv
assert '--no-container-mount-home' in sys.argv
assert any(x.startswith('--container-name=relh-generals-999') for x in sys.argv)
index = sys.argv.index('--')
os.execv(sys.argv[index+1], sys.argv[index+1:])
''',
            }
            for name, body in scripts.items():
                path = commands / name
                path.write_text(f"#!{sys.executable}\n" + body + "\n")
                path.chmod(0o700)
            worker = root / "worker.py"
            worker.write_text('''import pathlib,signal,sys,time
def emit(s):
    with open('events','a') as f: f.write(s+'\\n')
def stop(*args):
    emit('CHECKPOINT')
    time.sleep(.1)
    emit('LAST_WRITE')
    pathlib.Path('writing').unlink()
    emit('STOPPED')
    sys.exit(0)
signal.signal(signal.SIGTERM,stop)
pathlib.Path('writing').touch()
emit('READY')
if sys.argv[1]=='wait':
    while True: time.sleep(.01)
pathlib.Path('writing').unlink()
emit('STOPPED')
sys.exit(7 if sys.argv[1]=='fail' else 0)
''')
            config = dict(scratch_parent=str(root), receipt=dict(source="test"),
                          scratch_bytes=1, scratch_inodes=1, runtime_seconds=600,
                          credential_expiry=time.time()+3600,
                          input_url="https://input", input_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          input_unpacked_bytes=100, input_members=10,
                          enroot_unpack_path=str(root), image_unpacked_bytes=1, image_inodes=1,
                          image="test-image", output_urls=["https://part"], manifest_url="https://manifest",
                          steps=[dict(name="smoke", argv=["--", sys.executable, str(worker), "success"], seconds=5),
                                 dict(name="train", argv=["--", sys.executable, str(worker), mode], seconds=20)])
            (root / "config.json").write_text(json.dumps(config))
            harness = root / "harness.py"
            harness.write_text(f'''import json,pathlib,shutil,sys
from integrations import slurm_s3_job as m
def transfer(url,path,upload=False):
    if upload:
        assert not pathlib.Path('writing').exists()
        with open('events','a') as f: f.write('UPLOAD\\n')
        if {fail_upload!r}: raise RuntimeError('simulated upload failure')
        shutil.copyfile(path, 'uploaded-'+path.name)
    else: shutil.copyfile('input.tar.gz',path)
m.transfer=transfer
sys.exit(m.SlurmJob(json.load(open('config.json'))).execute())
''')
            env = dict(os.environ, SLURM_JOB_ID="999", PATH=str(commands)+os.pathsep+os.environ["PATH"],
                       PYTHONPATH=str(Path(__file__).resolve().parents[1]))
            process = subprocess.Popen([sys.executable, str(harness)], cwd=root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
            try:
                if sent_signal:
                    deadline = time.monotonic()+10
                    while not (root / "events").exists() or (root / "events").read_text().count("READY") < 2:
                        if time.monotonic() > deadline:
                            self.fail("Training child did not start")
                        time.sleep(.01)
                    process.send_signal(sent_signal)
                stdout, stderr = process.communicate(timeout=15)
                events = (root / "events").read_text().splitlines() if (root / "events").exists() else []
                result = dict(code=process.returncode, events=events, scratch=(root/"relh-generals-999").exists(),
                              cleaned=(root/"cleaned").exists(), stdout=stdout, stderr=stderr)
                manifest = root / "uploaded-result-manifest.json"
                if manifest.exists():
                    result["manifest"] = json.loads(manifest.read_text())
                return result
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()

    def test_success(self):
        r = self.run_job()
        self.assertEqual(r["code"], 0, r)
        self.assertTrue(r["cleaned"])
        self.assertFalse(r["scratch"])
        self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"], 0)

    def test_failed_training(self):
        r = self.run_job("fail")
        self.assertEqual(r["code"], 7, r)
        self.assertTrue(r["scratch"])
        self.assertFalse(r["cleaned"])
        self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"], 7)

    def test_signaled_training_finishes_writes_before_upload(self):
        for sig, code in ((signal.SIGUSR1,124),(signal.SIGTERM,143),(signal.SIGINT,130)):
            with self.subTest(signal=sig):
                r=self.run_job("wait",sig)
                self.assertEqual(r["code"],code,r)
                events=r["events"]
                self.assertLess(events.index("LAST_WRITE"),events.index("UPLOAD"))
                self.assertTrue(r["scratch"])
                self.assertFalse(r["cleaned"])
                self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"],code)

    def test_upload_failure(self):
        r=self.run_job(fail_upload=True)
        self.assertEqual(r["code"],74,r)
        self.assertTrue(r["scratch"])
        self.assertFalse(r["cleaned"])
        self.assertNotIn("manifest",r)

    def test_wrong_priority_never_downloads_or_starts(self):
        r=self.run_job(low_nice=True)
        self.assertEqual(r["code"],1,r)
        self.assertEqual(r["events"],[])
        self.assertFalse(r["scratch"])

    def test_existing_scratch_never_archived_or_removed(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, SLURM_JOB_ID="999"):
            job=SlurmJob(dict(scratch_parent=root,receipt={}))
            (job.root/"out").mkdir(parents=True)
            with patch.object(job,"check_priority"), patch.object(job,"archive_upload") as upload:
                self.assertEqual(job.execute(),1)
                upload.assert_not_called()
            self.assertTrue(job.root.exists())

    def test_input_rejects_links_traversal_and_oversize(self):
        for kind in ("link","traversal","oversize"):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                archive=root/"input.tar"
                with tarfile.open(archive,"w") as target:
                    member=tarfile.TarInfo("../escape" if kind=="traversal" else "file")
                    if kind=="link":
                        member.type=tarfile.SYMTYPE;member.linkname="/tmp"
                        target.addfile(member)
                    else:
                        member.size=4;target.addfile(member,io.BytesIO(b"data"))
                with self.assertRaises(ValueError):
                    extract_input(archive,root/"out",1 if kind=="oversize" else 10,10)
                self.assertFalse((root/"out").exists())


if __name__ == "__main__":
    unittest.main()

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
    def run_job(self, mode="success", sent_signal=None, fail_upload=False, low_nice=False,
                image_parts=False, bad_image=False, result_part_bytes=None, busy_gpu=False, gpu_query_failure=False, batch_gpu_absent=False,
                cleanup_failure=False, image_parallelism=1, lingering_reads=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / "bin"
            commands.mkdir()
            archive = root / "input.tar.gz"
            runtime = Path(__file__).resolve().parents[1] / 'integrations/slurm_s3_job.py'
            with tarfile.open(archive, "w:gz") as target:
                member = tarfile.TarInfo("source.txt")
                member.size = 4
                target.addfile(member, io.BytesIO(b"data"))
                target.add(runtime, arcname='source/integrations/slurm_s3_job.py')
                if not image_parts:
                    member = tarfile.TarInfo('image.sqsh')
                    member.size = 5
                    target.addfile(member, io.BytesIO(b'image'))
            scripts = {
                "scontrol": f"print('JobId=999 Nice={100 if low_nice else 2147483645} Priority=1 TimeLimit=00:10:00')",
                "squeue": f"""import pathlib
counter = pathlib.Path('queue-reads')
reads = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(reads + 1))
print('999.0' if pathlib.Path('writing').exists() or reads < {lingering_reads} else '')
""",
                "enroot": f'''import os,pathlib,sys,shutil
assert not pathlib.Path('writing').exists()
name='relh-generals-999'
data=pathlib.Path(os.environ['ENROOT_DATA_PATH'])
assert data.parent.parent.name == name
assert os.environ['ENROOT_MOUNT_HOME']=='no'
assert pathlib.Path(os.environ['ENROOT_CONFIG_PATH']).parent == data.parent
if sys.argv[1]=='create':
    assert sys.argv[2:4]==['--name',name]
    (data/name).mkdir()
elif sys.argv[1]=='start':
    assert os.environ['NVIDIA_VISIBLE_DEVICES']=='GPU-abcd'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-abcd'
    assert os.environ['NVIDIA_DRIVER_CAPABILITIES']=='compute,utility'
    index=sys.argv.index('--')
    assert sys.argv[index+1]==name
    assert sys.argv[index+2:index+6]==['/bin/sh','-c','cd /work && exec "$@"','generals-workload']
    command=sys.argv[index+6:]
    os.execv(command[0],command)
elif sys.argv[1]=='remove':
    assert sys.argv[1:]==['remove','--force',name]
    if {cleanup_failure!r}:
        print('owned root removal denied',file=sys.stderr); sys.exit(7)
    pathlib.Path('cleaned').touch()
    shutil.rmtree(data/name)
else:
    raise AssertionError('No global Enroot namespace listing permitted')
''',
                "nvidia-smi": f'''import sys,os
if {gpu_query_failure!r} or '--id=1' in sys.argv or ({batch_gpu_absent!r} and not os.environ.get('SLURM_STEP_GPUS')):
    print('No devices were found', file=sys.stderr); sys.exit(6)
if any('minor_number' in a for a in sys.argv): sys.exit(2)
if '--query-gpu=index,uuid' in sys.argv: print('0, GPU-abcd')
elif '--query-compute-apps=pid' in sys.argv: print({repr('123' if busy_gpu else '')})
else: print('GPU-abcd, 0, 0')
''',
                "srun": '''import os,sys
assert '--nice=2147483645' in sys.argv
assert '--gres=gpu:1' in sys.argv
os.environ['SLURM_STEP_GPUS']='1'
from integrations.slurm_s3_job import verify_allocated_gpu_idle
verify_allocated_gpu_idle()
assert not any(x.startswith('--container-') for x in sys.argv)
index = sys.argv.index('python3')
os.execv(sys.executable, sys.argv[index:])
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
            config = dict(scratch_parent=str(root), receipt=dict(source="test", source_hashes={
                              'integrations/slurm_s3_job.py':hashlib.sha256(runtime.read_bytes()).hexdigest()}),
                          scratch_bytes=1, scratch_inodes=1, runtime_seconds=600,
                          credential_expiry=time.time()+3600,
                          input_url="https://input", input_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          input_unpacked_bytes=100000, input_members=10,
                          image_unpacked_bytes=1, image_inodes=1,
                          workload_bytes=1, workload_inodes=1,
                          image="input/image.sqsh", image_sha256=hashlib.sha256(b'image').hexdigest(),
                          output_urls=["https://part"], manifest_url="https://manifest",
                          steps=[dict(name="smoke", argv=[sys.executable, str(worker), "success"], seconds=5),
                                 dict(name="train", argv=[sys.executable, str(worker), mode], seconds=20)])
            if image_parts:
                config["image_download_parallelism"] = image_parallelism
                config["image_parts"] = []
                for index, data in enumerate((b"first", b"second")):
                    (root / f"image{index}").write_bytes(data)
                    config["image_parts"].append(dict(url=f"https://image{index}",bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
                config["image_sha256"] = "0"*64 if bad_image else hashlib.sha256(b"firstsecond").hexdigest()
            if result_part_bytes:
                config["output_urls"] = [f"https://output{index}" for index in range(32)]
            (root / "config.json").write_text(json.dumps(config))
            harness = root / "harness.py"
            harness.write_text(f'''import json,pathlib,shutil,sys
from integrations import slurm_s3_job as m
def transfer(url,path,upload=False,deadline=None):
    if upload:
        assert not pathlib.Path('writing').exists()
        with open('events','a') as f: f.write('UPLOAD\\n')
        if {fail_upload!r}: raise RuntimeError('simulated upload failure')
        shutil.copyfile(path, 'uploaded-'+path.name)
    else: shutil.copyfile(url.removeprefix('https://') if url.startswith('https://image') else 'input.tar.gz',path)
m.transfer=transfer
m.download_parts=lambda parts: [transfer(url,path) for url,path in parts]
if {result_part_bytes!r}: m.MAX_PART_BYTES={result_part_bytes!r}
sys.exit(m.SlurmJob(json.load(open('config.json'))).execute())
''')
            env = dict(os.environ, SLURM_JOB_ID="999", SLURM_JOB_GPUS="1", PATH=str(commands)+os.pathsep+os.environ["PATH"],
                       ENROOT_DATA_PATH="/wrong-host-only-root", ENROOT_RUNTIME_PATH="/wrong-runtime",
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
                              cleaned=(root/"cleaned").exists(), stdout=stdout, stderr=stderr,
                              smoke_log=(root/"relh-generals-999/out/smoke.log").read_text() if (root/"relh-generals-999/out/smoke.log").exists() else "")
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

    def test_success_waits_for_lingering_remote_step(self):
        r = self.run_job(lingering_reads=2)
        self.assertEqual(r["code"], 0, r)
        self.assertEqual(r["events"].count("READY"), 2)
        self.assertTrue(r["cleaned"])
        self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"], 0)

    def test_unconfirmed_remote_step_wait_is_bounded(self):
        job = SlurmJob.__new__(SlurmJob)
        job.process = None
        job.create_process = None
        with patch.object(job, "steps_stopped", return_value=False) as stopped, \
                patch("integrations.slurm_s3_job.time.monotonic", side_effect=[0, 0, 46]), \
                patch("integrations.slurm_s3_job.time.sleep"):
            self.assertFalse(job.stop_and_wait())
        stopped.assert_called_once()

    def test_failed_training(self):
        r = self.run_job("fail")
        self.assertEqual(r["code"], 7, r)
        self.assertTrue(r["scratch"])
        self.assertTrue(r["cleaned"])
        self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"], 7)

    def test_cleanup_failure_preserves_uploaded_success_and_diagnostic(self):
        r = self.run_job(cleanup_failure=True)
        self.assertEqual(r['code'], 74, r)
        self.assertTrue(r['scratch'])
        self.assertFalse(r['cleaned'])
        self.assertEqual(r['manifest']['receipt']['workload_exit_code'], 0)
        self.assertIn('Owned container cleanup failed', r['manifest']['receipt']['container_cleanup_failure'])
        self.assertEqual(r['manifest']['receipt']['runner_exit_code'],74)

    def test_signaled_training_finishes_writes_before_upload(self):
        for sig, code in ((signal.SIGUSR1,124),(signal.SIGTERM,143),(signal.SIGINT,130)):
            with self.subTest(signal=sig):
                r=self.run_job("wait",sig)
                self.assertEqual(r["code"],code,r)
                events=r["events"]
                self.assertLess(events.index("LAST_WRITE"),events.index("UPLOAD"))
                self.assertTrue(r["scratch"])
                self.assertTrue(r["cleaned"])
                self.assertEqual(r["manifest"]["receipt"]["workload_exit_code"],code)

    def test_upload_failure(self):
        r=self.run_job(fail_upload=True)
        self.assertEqual(r["code"],74,r)
        self.assertTrue(r["scratch"])
        self.assertTrue(r["cleaned"])
        self.assertNotIn("manifest",r)

    def test_wrong_priority_never_downloads_or_starts(self):
        r=self.run_job(low_nice=True)
        self.assertEqual(r["code"],1,r)
        self.assertEqual(r["events"],[])
        self.assertFalse(r["scratch"])

    def test_image_parts_verified_before_any_step(self):
        good=self.run_job(image_parts=True)
        self.assertEqual(good["code"],0,good)
        bad=self.run_job(image_parts=True,bad_image=True)
        self.assertEqual(bad["code"],1,bad)
        self.assertNotIn("READY",bad["events"])
        self.assertTrue(bad["scratch"])

    def test_parallel_image_parts_preserve_hash_gate_and_receipt(self):
        good = self.run_job(image_parts=True, image_parallelism=4)
        self.assertEqual(good["code"], 0, good)
        self.assertEqual(good["manifest"]["receipt"]["image_download"]["bytes"], 11)
        self.assertEqual(good["manifest"]["receipt"]["image_download"]["parallelism"], 4)
        bad = self.run_job(image_parts=True, image_parallelism=4, bad_image=True)
        self.assertEqual(bad["code"], 1, bad)
        self.assertNotIn("READY", bad["events"])

    def test_invalid_image_parallelism_never_starts_step(self):
        for value in (0, 5, True, 1.5):
            with self.subTest(parallelism=value):
                result = self.run_job(image_parts=True, image_parallelism=value)
                self.assertEqual(result["code"], 1, result)
                self.assertNotIn("READY", result["events"])

    def test_busy_physical_gpu_prevents_workload(self):
        result=self.run_job(busy_gpu=True)
        self.assertEqual(result["code"],1,result)
        self.assertNotIn("READY",result["events"])
        self.assertTrue(result["cleaned"])

    def test_global_gpu_ordinal_is_resolved_to_visible_uuid(self):
        # Fake NVML exposes physical minor1 as visible index0. -i1 returns6,
        # reproducing the old preparation failure without a GPU allocation.
        result = self.run_job()
        self.assertEqual(result["code"], 0, result)
        self.assertEqual(result["manifest"]["receipt"]["batch_gpu_visibility"], "0, GPU-abcd")

    def test_batch_without_devices_does_not_skip_step_gpu_gate(self):
        result = self.run_job(batch_gpu_absent=True)
        self.assertEqual(result["code"], 0, result)
        self.assertIn("query exit 6", result["manifest"]["receipt"]["batch_gpu_visibility"])
        busy = self.run_job(batch_gpu_absent=True, busy_gpu=True)
        self.assertEqual(busy["code"], 1, busy)
        self.assertNotIn("READY", busy["events"])

    def test_gpu_query_failure_retains_diagnostic_and_never_starts(self):
        result = self.run_job(gpu_query_failure=True)
        self.assertEqual(result["code"], 1, result)
        self.assertNotIn("READY", result["events"])
        failure = result["smoke_log"]
        self.assertIn("query exit 6", failure)
        self.assertIn("No devices were found", failure)

    def test_multiple_output_parts_have_completion_manifest(self):
        result=self.run_job(result_part_bytes=128)
        self.assertEqual(result["code"],0,result)
        self.assertGreater(len(result["manifest"]["parts"]),1)
        self.assertTrue(all(part["bytes"]<=128 for part in result["manifest"]["parts"]))

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


def test_storage_failure_records_exact_available_and_required_values(tmp_path):
    from types import SimpleNamespace
    import pytest
    from integrations.slurm_s3_job import check_space
    receipt = {}
    with patch('os.statvfs', return_value=SimpleNamespace(f_bavail=1000, f_frsize=4096, f_favail=13)):
        with pytest.raises(RuntimeError, match='available_inodes.*13') as error:
            check_space(tmp_path, 1024, 60, receipt=receipt)
    expected = dict(path=str(tmp_path.resolve()), available_bytes=4096000, required_bytes=1024,
                    available_inodes=13, required_inodes=60)
    assert receipt['storage_checks'] == [expected]
    assert json.loads(str(error.value).split(': ', 1)[1]) == expected


def test_retired_recovery_mount_configuration_is_rejected():
    import pytest
    with patch.dict(os.environ, {'SLURM_JOB_ID': '999'}):
        with pytest.raises(ValueError, match='mount_recovery'):
            SlurmJob(dict(scratch_parent='/tmp', receipt={}, mount_recovery=True))


def test_timed_out_container_creation_stops_descendant_writer_before_archive(tmp_path):
    import pytest
    from integrations.slurm_s3_job import SlurmJob
    program = '''import os,pathlib,signal,sys,time
root=pathlib.Path(sys.argv[1])
if os.fork()==0:
    def stop(*args):
        time.sleep(.1)
        (root/'last-write').write_text('finished')
        (root/'writing').unlink()
        os._exit(0)
    signal.signal(signal.SIGTERM,stop)
    (root/'writing').touch()
    while True: time.sleep(.01)
while True: time.sleep(.01)
'''
    with patch.dict(os.environ, {'SLURM_JOB_ID':'999'}):
        job=SlurmJob(dict(scratch_parent=str(tmp_path),receipt={}))
    job.create_process=subprocess.Popen([sys.executable,'-c',program,str(tmp_path)],start_new_session=True)
    process=job.create_process
    try:
        deadline=time.monotonic()+3
        while not (tmp_path/'writing').exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=.05)
        assert job.stop_and_wait()
        assert (tmp_path/'last-write').read_text()=='finished'
        assert not (tmp_path/'writing').exists()
        assert job.create_process is None
        assert job.receipt['container_creation_stopped'] is True
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=3)

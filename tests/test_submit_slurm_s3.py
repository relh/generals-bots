import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from integrations.submit_slurm_s3 import render, submit, monitor


class SubmissionTests(unittest.TestCase):
    def config(self):
        return dict(runtime_seconds=3300, credential_expiry=time.time()+36000,
                    enroot_storage_paths=["/tmp"],
                    required_gpu_memory_gib=198, steps=[dict(name="smoke", seconds=60)],
                    input_url="https://example.invalid/credential-SECRET")

    def test_rendered_script_is_valid_and_lowest_priority(self):
        script=render(self.config(),name="relh-generals-test",partition="b300",minutes=55,cpus=8,memory_gib=64)
        self.assertIn("#SBATCH --nice=2147483645",script)
        self.assertIn("#SBATCH --no-requeue",script)
        self.assertIn("#SBATCH --signal=B:USR1@600",script)
        self.assertNotIn("--nodelist",script)
        result=subprocess.run(["bash","-n"],input=script,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        body=script.split("<<'GENERALS_HOST_RUNNER'\n",1)[1].rsplit("\nGENERALS_HOST_RUNNER",1)[0]
        compile(body,"rendered-host-runner","exec")

    def test_rejects_ineligible_resources_and_expiring_credentials(self):
        for override in (dict(required_gpu_memory_gib=20),dict(credential_expiry=time.time()+100),
                         dict(enroot_storage_paths=[]), dict(enroot_storage_paths=["relative"]),
                         dict(steps=[dict(name="smoke",seconds=3000)])):
            with self.subTest(override=override),self.assertRaises(ValueError):
                render(self.config()|override,name="relh-generals-test",partition="b300",minutes=55,cpus=8,memory_gib=64)

    def test_live_task_prevents_submission(self):
        result=subprocess.CompletedProcess([],0,"35687|relh-classic-old\n","")
        with tempfile.TemporaryDirectory() as directory,patch("integrations.submit_slurm_s3.remote",return_value=result) as remote:
            receipt=Path(directory)/"receipt.json"
            with self.assertRaises(RuntimeError): submit("secret script",receipt,{},"metta0")
            self.assertFalse(receipt.exists())
            self.assertEqual(remote.call_count,1)

    def test_lost_response_preserves_intent_without_retry(self):
        responses=[subprocess.CompletedProcess([],0,"", ""),subprocess.TimeoutExpired("ssh",45)]
        with tempfile.TemporaryDirectory() as directory,patch("integrations.submit_slurm_s3.remote",side_effect=responses) as remote:
            receipt=Path(directory)/"receipt.json"
            with self.assertRaises(subprocess.TimeoutExpired): submit("secret script",receipt,{},"metta0")
            self.assertEqual(json.loads(receipt.read_text())["state"],"SUBMISSION_INTENT")
            self.assertEqual(remote.call_count,2)

    def test_submission_and_terminal_receipts_exclude_credentials(self):
        def record(state):
            return subprocess.CompletedProcess([],0,f"JobId=999 Nice=2147483645 Priority=1 JobState={state} ExitCode=0:0 SubmitLine=SECRET_URL\n","")
        responses=[subprocess.CompletedProcess([],0,"", ""),subprocess.CompletedProcess([],0,"999\n",""),record("PENDING"),record("COMPLETED")]
        with tempfile.TemporaryDirectory() as directory,patch("integrations.submit_slurm_s3.remote",side_effect=responses):
            receipt=Path(directory)/"receipt.json"
            job=submit("secret script",receipt,{"source":"test","s3_key":"relh/test"},"metta0")
            monitor("metta0",job,receipt)
            self.assertEqual(json.loads(receipt.read_text())["JobState"],"COMPLETED")
            self.assertNotIn("SECRET",receipt.read_text())


if __name__ == "__main__": unittest.main()

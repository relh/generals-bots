"""Retire only an inactive owned image root after matching preserved results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from integrations.slurm_s3_job import SlurmJob


class RetainedContainerTests(unittest.TestCase):
    def exercise(self, *, active=False, steps=False, corrupt=False, wrong_owner=False,
                 wrong_receipt=False, symlink=False, remove_failure=False):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            old = parent / 'relh-generals-111'
            (old / 'out').mkdir(parents=True)
            archive = old / 'results.tar.gz'
            archive.write_bytes(b'verified and preserved results')
            (old / 'out/receipt.json').write_text(json.dumps({'job_id': '112' if wrong_receipt else '111'}))
            expected = hashlib.sha256(archive.read_bytes()).hexdigest()
            if symlink:
                archive.rename(old / 'original.tar.gz')
                archive.symlink_to(old / 'original.tar.gz')
            job = SlurmJob.__new__(SlurmJob)
            job.job_id = '222'
            job.root = parent / 'relh-generals-222'
            job.runtime_env = {}
            job.receipt = {}
            job.config = {'retained_container': {'job_id': '111', 'results_sha256': '0' * 64 if corrupt else expected}}
            original = {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()}
            containers = {'pyxis_relh-generals-111', 'pyxis_relh-generals-999', 'unrelated'}
            removed = []

            def run(command, **kwargs):
                if command[0] == 'squeue':
                    stdout = '999\n'
                    if active and ('--steps' in command) == steps:
                        stdout += '111.0\n' if steps else '111\n'
                    return subprocess.CompletedProcess(command, 0, stdout, '')
                if command == ['enroot', 'list']:
                    return subprocess.CompletedProcess(command, 0, '\n'.join(sorted(containers)), '')
                self.assertEqual(command, ['enroot', 'remove', '--force', 'pyxis_relh-generals-111'])
                if remove_failure:
                    return subprocess.CompletedProcess(command, 7, '', 'denied')
                removed.append(command[-1])
                containers.remove(command[-1])
                return subprocess.CompletedProcess(command, 0, '', '')

            uid = os.getuid()
            with patch('integrations.slurm_s3_job.subprocess.run', side_effect=run), \
                    patch('integrations.slurm_s3_job.os.getuid', return_value=uid + 1 if wrong_owner else uid):
                if any((active, corrupt, wrong_owner, wrong_receipt, symlink, remove_failure)):
                    with self.assertRaises((RuntimeError, ValueError)):
                        job.retire_verified_container()
                    self.assertEqual(removed, [])
                else:
                    job.retire_verified_container()
                    self.assertEqual(removed, ['pyxis_relh-generals-111'])
                    self.assertTrue(job.receipt['retired_owned_container']['source_and_results_preserved'])
            self.assertEqual(original, {str(p.relative_to(old)): p.read_bytes() for p in old.rglob('*') if p.is_file()})
            self.assertIn('pyxis_relh-generals-999', containers)
            self.assertIn('unrelated', containers)

    def test_retire_only_verified_owned_container_preserves_every_file(self):
        self.exercise()

    def test_refuse_active_job_or_step(self):
        for steps in (False, True):
            with self.subTest(steps=steps):
                self.exercise(active=True, steps=steps)

    def test_refuse_unverified_identity(self):
        for field in ('corrupt', 'wrong_owner', 'wrong_receipt', 'symlink'):
            with self.subTest(field=field):
                self.exercise(**{field: True})

    def test_remove_failure_preserves_data(self):
        self.exercise(remove_failure=True)

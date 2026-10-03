import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from integrations import portable_classic_pilot as pilot


class PilotGateTests(unittest.TestCase):
    def test_evaluation_child_receives_complete_optimizer_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            original = subprocess.Popen
            def launch(_argv, **kwargs):
                code = ('import os; '
                        'assert os.environ["METTA_SPATIAL_OPTIMIZER_LAYOUT"] == "logical"; '
                        'assert os.environ["METTA_SPATIAL_MUON_DENSE_ORIENTATION"] == "canonical"; '
                        'assert os.environ["METTA_SPATIAL_MUON_CONTEXT_MATRIX"] == "1"')
                return original([sys.executable, '-c', code], **kwargs)
            with patch.object(pilot, 'OUT', Path(directory)), patch.object(
                    pilot.subprocess, 'Popen', side_effect=launch):
                pilot.command('audit_spatial_checkpoint_serving_parity', name='parity', seconds=5)

    def test_occupied_gpu_stops_before_jax_or_config_creation(self):
        jax = SimpleNamespace(devices=Mock())
        with patch.dict('sys.modules', jax=jax), patch(
                'integrations.slurm_s3_job.verify_allocated_gpu_idle',
                side_effect=RuntimeError('occupied')), patch.object(pilot, 'prepare_configs') as prepare:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                pilot.smoke()
            jax.devices.assert_not_called()
            prepare.assert_not_called()

    def test_missing_native_adapter_terminates_worker_before_an_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'run').mkdir()
            console = output/'run/console.log'
            original = subprocess.Popen
            children = []
            def launch(_argv, **kwargs):
                script = ("from pathlib import Path; import time; "
                          + f"Path({str(console)!r}).write_text('pooling: generic Fabric setup\\n'); "
                          + "time.sleep(30)")
                child = original([sys.executable, '-c', script], **kwargs)
                children.append(child)
                return child
            with patch.object(pilot, 'OUT', output), patch.object(pilot, 'allocated_gpu', return_value='GPU-abcd'), patch.object(
                    pilot.subprocess, 'run'), patch.object(pilot.subprocess, 'Popen', side_effect=launch):
                try:
                    with self.assertRaisesRegex(RuntimeError, 'without the verified spatial adapter'):
                        pilot.command('unused', name='native', train=True, seconds=5)
                    self.assertLess(children[0].wait(timeout=3), 0)
                finally:
                    for child in children:
                        if child.poll() is None:
                            child.kill()
                            child.wait(timeout=3)

    def test_success_records_uuid_before_jax(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            identity = dict(uuid='GPU-abcd', visible_index='0', slurm_assignment='1')
            def devices(platform):
                self.assertEqual(json.loads((output/'gpu-preflight.json').read_text()), identity)
                return [object()]
            jax = SimpleNamespace(devices=devices, __version__='cpu-fixture')
            original = Path.read_text
            def read(path, *args, **kwargs):
                return '{}' if str(path) == '/work/input/source-manifest.json' else original(path, *args, **kwargs)
            with patch.dict('sys.modules', jax=jax), patch.object(pilot, 'OUT', output), patch(
                    'integrations.slurm_s3_job.verify_allocated_gpu_idle', return_value=identity), patch.object(
                    pilot, 'allocated_gpu', return_value=identity['uuid']), patch.object(
                    pilot.subprocess, 'run'), patch.object(pilot, 'prepare_configs'), patch.object(Path, 'read_text', read):
                pilot.smoke()


if __name__ == '__main__': unittest.main()

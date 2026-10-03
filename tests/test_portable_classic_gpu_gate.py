import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from integrations import portable_classic_pilot as pilot


class PilotGateTests(unittest.TestCase):
    def test_occupied_gpu_stops_before_jax_or_config_creation(self):
        jax = SimpleNamespace(devices=Mock())
        with patch.dict('sys.modules', jax=jax), patch(
                'integrations.slurm_s3_job.verify_allocated_gpu_idle',
                side_effect=RuntimeError('occupied')), patch.object(pilot, 'prepare_configs') as prepare:
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                pilot.smoke()
            jax.devices.assert_not_called()
            prepare.assert_not_called()

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

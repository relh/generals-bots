import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / 'integrations/puffer_bootstrap'
CODE = 'from integrations.launch_spatial_selfplay_training import verify_runtime_bootstrap; verify_runtime_bootstrap(); print("VERIFIED")'


class BootstrapTests(unittest.TestCase):
    def env(self, path):
        return dict(os.environ, PYTHONPATH=path, METTA_MEMORYLESS_OPTIMIZATION='1',
                    METTA_AUDIT_DEVICE_REWARDS='0', METTA_SPATIAL_BOOTSTRAP_PID='stale-parent')

    def test_flag_alone_and_inherited_marker_fail(self):
        r = subprocess.run([sys.executable, '-c', CODE], env=self.env(str(ROOT)),
                           capture_output=True, text=True, timeout=20)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('bootstrap in this interpreter', r.stderr)

    def test_fresh_parent_and_child_each_activate(self):
        code = CODE + '; import subprocess,sys; subprocess.run([sys.executable,"-c",' + repr(CODE) + '],check=True)'
        r = subprocess.run([sys.executable, '-c', code], env=self.env(f'{BOOT}:{ROOT}'),
                           capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count('VERIFIED'), 2)
        pids = [line for line in r.stdout.splitlines() if line.startswith('SPATIAL_BOOTSTRAP_READY')]
        self.assertEqual(len(set(pids)), 2)

    def test_bootstrap_import_failure_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)/'integrations'; package.mkdir()
            (package/'__init__.py').touch()
            # Disable installed editable import hooks so this dependency failure
            # cannot accidentally resolve to the real training adapter.
            code = f'import runpy; runpy.run_path({str(BOOT / "sitecustomize.py")!r}); print("UNSAFE_START")'
            r = subprocess.run([sys.executable, '-S', '-c', code],
                               cwd=directory, env=self.env(f'{BOOT}:{directory}'),
                               capture_output=True, text=True, timeout=20)
            self.assertEqual(r.returncode, 78)
            self.assertNotIn('UNSAFE_START', r.stdout)


if __name__ == '__main__': unittest.main()

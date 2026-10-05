import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations.cuda_runtime_binding import configure, library_path


class CudaBindingTests(unittest.TestCase):
    def test_requires_all_wheel_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'missing'):
                library_path('/usr/local/cuda/lib64', Path(directory))

    def test_reexec_preserves_module_arguments_and_precedes_base_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('cu13', 'cudnn', 'nccl'):
                (root / 'nvidia' / name / 'lib').mkdir(parents=True)
            argv = ['python', '-m', 'integrations.policy_trial', 'control']
            with patch.dict(os.environ, {'LD_LIBRARY_PATH': '/usr/local/cuda/lib64'}, clear=True), \
                    patch('sysconfig.get_path', return_value=directory), \
                    patch('sys.orig_argv', argv), \
                    patch('os.execvpe', side_effect=RuntimeError('reexec')) as execute:
                with self.assertRaisesRegex(RuntimeError, 'reexec'):
                    configure()
                binary, actual, env = execute.call_args.args
                self.assertEqual(actual, [binary, *argv[1:]])
                self.assertTrue(env['LD_LIBRARY_PATH'].startswith(str(root / 'nvidia/cu13/lib') + ':'))
                self.assertTrue(env['LD_LIBRARY_PATH'].endswith('/usr/local/cuda/lib64'))
                # The re-executed interpreter must not loop, and children inherit the audit flag.
                with patch.dict(os.environ, env, clear=True):
                    execute.reset_mock()
                    configure()
                    execute.assert_not_called()
                    self.assertEqual(os.environ['GENERALS_AUDIT_CUDA_RUNTIME'], '1')


if __name__ == '__main__':
    unittest.main()

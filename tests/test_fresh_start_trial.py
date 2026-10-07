import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from integrations import fresh_start_run as run
from integrations.row_rotation_trial import Trial, PLAN, read, write


class FreshStartAdmission(unittest.TestCase):
    def test_pending_plan_rejected(self):
        pending = read(run.PLAN_PATH)
        pending['status'] = 'pending evidence bindings'
        with patch.object(run, 'read', return_value=pending):
            with self.assertRaisesRegex(ValueError, 'pending evidence'):
                run.bound_plan(Path('/nonexistent'))

    def test_only_distribution_scalar_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = read(run.CONFIG/'build-config.json')
            write(root/'control/candidate/build-config.json', reference)
            build = copy.deepcopy(reference)
            build['python_environment']['options']['coworld_position_probability'] = 0.0
            write(root/'build-config.json', build)
            write(root/'config.json', read(run.CONFIG/'training-config.json'))
            self.assertEqual(run.configs(root)[0], build)
            build['python_environment']['options']['classic_siege_workers'] = 4
            write(root/'build-config.json', build)
            with self.assertRaisesRegex(ValueError, 'Only the fixed'):
                run.configs(root)

    def test_all_rolling_windows_reject_hidden_slow_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'run').mkdir()
            def log(times):
                (root/'run/console.log').write_text('INITIAL_POSITION_MIX {"games":4096,"midgame":0,"probability":0.0}\n' + ''.join(
                    f'╭ Epoch {epoch} Uptime {seconds}s\n' for epoch, seconds in enumerate(times, 1)))
            log([100, 112, 124, 136, 148, 160, 172, 184])
            run.intervals(root, 1, 8)
            # Full interval still exceeds 30K; middle two epochs do not.
            log([100, 112, 113, 114, 150, 151, 152, 153])
            with self.assertRaisesRegex(ValueError, 'rolling'):
                run.intervals(root, 1, 8)

    def test_trial_plan_is_instance_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root/'assets/cold/asset.json', dict(sampler={}))
            custom = copy.deepcopy(PLAN)
            custom['evaluation_seed'] = 17001101
            self.assertEqual(Trial(root, root/'a', plan=custom).plan['evaluation_seed'], 17001101)
            self.assertEqual(Trial(root, root/'b').plan['evaluation_seed'], 14001101)

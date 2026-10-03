from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations import portable_classic_pilot as pilot


class LearningCurveTests(unittest.TestCase):
    def test_budget_is_bounded(self):
        self.assertEqual(pilot.pilot_steps("33554432"), 33_554_432)
        for invalid in ("0", "1000000000", "-1", "33554433"):
            with self.assertRaises(ValueError):
                pilot.pilot_steps(invalid)

    def test_mid_and_final_share_one_matched_parent_panel(self):
        with patch.object(pilot, 'STEPS', 33_554_432), patch.object(
                pilot, 'export_and_audit') as export, patch.object(pilot, 'command') as command:
            pilot.evaluate()
        self.assertEqual([call.args[0] for call in export.call_args_list], [33_554_432, 16_777_216])
        panels = [call for call in command.call_args_list if call.args[0] == 'evaluate_spatial_population']
        self.assertEqual([call.kwargs['name'] for call in panels], ['heldout-parent', 'heldout-child', 'heldout-mid'])
        for call in panels:
            for key, expected in (('--games', 4096), ('--pool-size', 4096), ('--seed', 37813), ('--sample-seed', 8713)):
                self.assertEqual(call.args[call.args.index(key) + 1], expected)
        pairs = [call for call in command.call_args_list if call.args[0] == 'analyze_spatial_population_pair']
        self.assertEqual([call.kwargs['name'] for call in pairs], ['paired-child', 'paired-mid'])

    def test_missing_intermediate_checkpoint_cannot_silently_use_final(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, 'OUT', Path(directory)), patch.object(
                pilot, 'command') as command:
            with self.assertRaises(FileNotFoundError):
                pilot.export_and_audit(16_777_216, Path(directory) / 'bundle-mid', '-mid')
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()

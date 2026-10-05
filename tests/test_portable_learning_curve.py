import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations import portable_classic_pilot as pilot


class LearningCurveTests(unittest.TestCase):
    def test_branching_from_same_parent_can_use_fresh_matched_seeds(self):
        with patch.dict(pilot.os.environ, GENERALS_EVAL_SEED="40913", GENERALS_EVAL_SAMPLE_SEED="10211"):
            self.assertEqual(pilot.evaluation_seeds(), (40913, 10211))
        with patch.dict(pilot.os.environ, {"GENERALS_EVAL_SEED": "40913"}, clear=True):
            with self.assertRaisesRegex(ValueError, 'paired'):
                pilot.evaluation_seeds()
        with patch.dict(pilot.os.environ, GENERALS_EVAL_SEED="-1", GENERALS_EVAL_SAMPLE_SEED="10211"):
            with self.assertRaisesRegex(ValueError, 'uint32'):
                pilot.evaluation_seeds()

    def test_budget_is_bounded(self):
        self.assertEqual(pilot.pilot_steps("33554432"), 33_554_432)
        self.assertEqual(pilot.pilot_steps("268435456"), 268_435_456)
        for invalid in ("0", "1000000000", "-1", "33554433"):
            with self.assertRaises(ValueError):
                pilot.pilot_steps(invalid)

    def test_mid_and_final_share_one_matched_parent_panel(self):
        with patch.object(pilot, 'STEPS', 33_554_432), patch.object(
                pilot, 'midpoint_checkpoint_steps', return_value=16_777_216), patch.object(
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

    def test_exploration_retains_qualified_parent_and_actual_initialization(self):
        with patch.dict(pilot.TRAIN_ENV, METTA_SPATIAL_FULL_ACTION_TEMPERATURE="10"), patch.object(
                pilot, 'STEPS', 33_554_432), patch.object(pilot, 'starting_steps', return_value=0), patch.object(
                pilot, 'midpoint_checkpoint_steps', return_value=16_777_216), patch.object(
                pilot, 'exploratory_initialization', return_value=pilot.OUT / 'exploratory-initialization'), patch.object(
                pilot, 'export_and_audit'), patch.object(pilot, 'command') as command:
            pilot.evaluate()
        panels = {call.kwargs['name']: call for call in command.call_args_list
                  if call.args[0] == 'evaluate_spatial_population'}
        self.assertEqual(set(panels), {'heldout-parent', 'heldout-child', 'heldout-mid', 'heldout-cold-parent'})
        for name, bundle in [('heldout-parent', 'exploratory-initialization'), ('heldout-cold-parent', 'self_bundle')]:
            call = panels[name]
            self.assertEqual(call.args[call.args.index('--bundle') + 1], pilot.OUT / bundle)
        pairs = {call.kwargs['name']: call for call in command.call_args_list
                 if call.args[0] == 'analyze_spatial_population_pair'}
        for name in ('paired-cold-child', 'paired-cold-mid'):
            self.assertEqual(pairs[name].args[pairs[name].args.index('--baseline') + 1],
                             pilot.OUT / 'heldout-cold-parent')

    def test_missing_intermediate_checkpoint_cannot_silently_use_final(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, 'OUT', Path(directory)), patch.object(
                pilot, 'command') as command:
            with self.assertRaises(FileNotFoundError):
                pilot.export_and_audit(16_777_216, Path(directory) / 'bundle-mid', '-mid')
            command.assert_not_called()

    def test_extended_run_keeps_midpoint_and_fresh_matched_panels(self):
        with patch.object(pilot, 'STEPS', 268_435_456), patch.object(
                pilot, 'midpoint_checkpoint_steps', return_value=134_217_728), patch.object(
                pilot, 'export_and_audit') as export, patch.object(pilot, 'command') as command:
            pilot.evaluate()
        self.assertEqual([call.args[0] for call in export.call_args_list], [268_435_456, 134_217_728])
        panels = [call for call in command.call_args_list if call.args[0] == 'evaluate_spatial_population']
        self.assertEqual(len(panels), 3)
        for call in panels:
            self.assertEqual(call.args[call.args.index('--games') + 1], 4096)
            self.assertEqual(call.args[call.args.index('--seed') + 1], 37841)
            self.assertEqual(call.args[call.args.index('--sample-seed') + 1], 8729)

    def midpoint_fixture(self, directory, epochs, interval=8):
        root = Path(directory)
        checkpoints = root / 'run/checkpoints/metta_generals/run'
        checkpoints.mkdir(parents=True)
        for epoch in epochs:
            (checkpoints / f'{epoch * 2097152:016d}.bin').write_bytes(b'fixture')
        (root / 'config.json').write_text(json.dumps({'overrides': {'base.checkpoint_interval': interval, 'vec.total_agents': 8192, 'train.horizon': 256, 'train.minibatch_size': 8192}}))
        return root

    def test_resumed_pilot_midpoint_between_save_boundaries(self):
        # Actual35867: resume644, requested midpoint708, checkpoints704/712.
        with tempfile.TemporaryDirectory() as directory:
            root = self.midpoint_fixture(directory, [640, 648, 704, 712, 772])
            with patch.object(pilot, 'OUT', root):
                selected = pilot.midpoint_checkpoint_steps(644 * 2097152, 128 * 2097152)
            self.assertEqual(selected, 704 * 2097152)
            record = json.loads((root / 'midpoint-selection.json').read_text())
            self.assertEqual(record['requested_agent_steps'], 708 * 2097152)
            self.assertEqual(record['offset_steps'], -4 * 2097152)

    def test_exact_saved_midpoint_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.midpoint_fixture(directory, [640, 696, 704, 712, 768])
            with patch.object(pilot, 'OUT', root):
                self.assertEqual(pilot.midpoint_checkpoint_steps(640 * 2097152, 128 * 2097152), 704 * 2097152)

    def test_missing_midpoint_does_not_use_parent_final_or_distant_snapshot(self):
        for epochs in ([644, 772], [644, 648, 772]):
            with self.subTest(epochs=epochs), tempfile.TemporaryDirectory() as directory:
                root = self.midpoint_fixture(directory, epochs)
                with patch.object(pilot, 'OUT', root), self.assertRaises(FileNotFoundError):
                    pilot.midpoint_checkpoint_steps(644 * 2097152, 128 * 2097152)
                self.assertFalse((root / 'midpoint-selection.json').exists())


if __name__ == '__main__':
    unittest.main()

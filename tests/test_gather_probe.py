import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from integrations import gather_probe_run as probe
from integrations.gather_probe_entrypoint import completion


class GatherProbeTest(unittest.TestCase):
    def test_pending_plan_cannot_launch(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                probe.prepare(Path(directory), Path(directory)/'out')

    def test_changed_diagnostic_budget_rejected(self):
        plan = json.loads(probe.PLAN.read_text())
        plan.update(status='sealed before launch', per_run_steps=4194304)
        with patch.object(probe, 'read', return_value=plan):
            with self.assertRaisesRegex(ValueError, 'Unsealed'):
                probe.prepare(Path('/missing'), Path('/unused'))

    def test_completion_never_qualifies_long_training(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            probe.write(root/'plan.json', {})
            probe.write(root/'COMPLETED.json', dict(plan_sha256=probe.digest(root/'plan.json'),
                                                   qualified_for_long_training=True))
            with self.assertRaisesRegex(ValueError, 'binding'):
                completion(root)

    def test_completion_does_not_duplicate_archive_plan_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            probe.write(root/'plan.json', {})
            probe.write(root/'COMPLETED.json', dict(plan_sha256=probe.digest(root/'plan.json'),
                                                   qualified_for_long_training=False))
            self.assertEqual(completion(root), {'qualified_for_long_training': False})

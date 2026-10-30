import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from integrations.recover_spatial_population_panel import recovery_plan


class PopulationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.reference, self.interrupted, self.bundle = [root / n for n in ('reference', 'interrupted', 'bundle')]
        for path in (self.reference, self.interrupted, self.bundle):
            path.mkdir()
        self.build = root / 'build.json'
        self.build.write_text('{}')
        (self.bundle / 'policy.bin').write_bytes(b'checkpoint')
        (self.bundle / 'spatial-policy.json').write_text('{}')
        self.output = root / 'recovered'
        self.digests = [hashlib.sha256((self.bundle / n).read_bytes()).hexdigest()
                        for n in ('policy.bin', 'spatial-policy.json')]
        arrays = dict(initial_state_sha256=np.array(['map-a', 'map-b']),
                      initial_sides=np.array([0, 1]), opponent_labels=np.array([0, 0]))
        for name, value in arrays.items():
            for directory in (self.reference, self.interrupted):
                np.save(directory / (name + '.npy'), value)
        np.save(self.reference / 'outcomes.npy', np.array([1, -1]))
        record = dict(games=2, pool_size=2, seed=71, sample_seed=89,
                      population_build_sha256=hashlib.sha256(self.build.read_bytes()).hexdigest(),
                      frozen_policy_sha256=['opponent'], opponent_weights=[1],
                      coworld_classic_rules=True, checkpoint_sha256='reference-policy',
                      wins=1, losses=1, draws=0, by_opponent_and_seat={'opponent': {
                          '0': dict(games=1, wins=1, losses=0, draws=0),
                          '1': dict(games=1, wins=0, losses=1, draws=0)}})
        (self.reference / 'evaluation.json').write_text(json.dumps(record))
        self.addCleanup(patch.stopall)
        patch('integrations.recover_spatial_population_panel.SpatialPlayerPolicy',
              return_value=SimpleNamespace(action_mode='structured_sample')).start()

    def plan(self):
        return recovery_plan(self.reference, self.interrupted, self.bundle,
                             self.build, self.output, *self.digests)

    def test_plan_preserves_inputs_and_reuses_exact_matched_settings(self):
        before = {p: p.read_bytes() for p in Path(self.temp.name).rglob('*') if p.is_file()}
        plan = self.plan()
        argv = plan['arguments']
        for flag, value in (('--games', '2'), ('--pool-size', '2'), ('--seed', '71'), ('--sample-seed', '89')):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {p: p.read_bytes() for p in Path(self.temp.name).rglob('*') if p.is_file()})

    def test_mismatched_initial_states_fail_before_execution(self):
        np.save(self.interrupted / 'initial_state_sha256.npy', np.array(['other', 'map-b']))
        with self.assertRaisesRegex(ValueError, 'initial_state_sha256'):
            self.plan()

    def test_changed_population_and_sampler_manifest_rejected(self):
        self.build.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'Population build'):
            self.plan()
        self.build.write_text('{}')
        (self.bundle / 'spatial-policy.json').write_text('{"temperature":2}')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            self.plan()

    def test_wrong_checkpoint_is_rejected(self):
        (self.bundle / 'policy.bin').write_bytes(b'another checkpoint')
        with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
            self.plan()

    def test_completed_or_partial_output_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.plan()
        self.output.rmdir()
        np.save(self.interrupted / 'outcomes.npy', np.array([1, -1]))
        with self.assertRaisesRegex(ValueError, 'contains outcomes'):
            self.plan()

    def test_invalid_completed_reference_is_not_trusted(self):
        np.save(self.reference / 'outcomes.npy', np.array([-1, -1]))
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            self.plan()

    def test_recovery_cannot_write_inside_retained_input(self):
        self.output = self.interrupted / 'new'
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.plan()


if __name__ == '__main__':
    unittest.main()

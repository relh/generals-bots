import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations import portable_classic_pilot as pilot
from integrations.classic_learner_continuation import load_continuation


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.parent = self.root / "parent"
        self.steps = 268_435_456
        self.checkpoint = self.parent / f"run/checkpoints/metta_generals/run/{self.steps:016d}.bin"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
        (self.parent / "bundle").mkdir()
        (self.parent / "bundle/policy.bin").write_bytes(self.checkpoint.read_bytes())
        self.state = Path(str(self.checkpoint) + ".learner")
        self.state.write_bytes(struct.pack("<8sQQQf4f", b"METTAL01", 128, self.steps, 4, 0.0002, 1, 2, 3, 4))
        record = dict(
            build=dict(
                config=dict(
                    python_environment=dict(
                        options=dict(
                            reward_scale=0.5,
                            terminal_reward_mode="win_only",
                            shaping_gamma=0.999,
                            coworld_classic=True,
                            teacher=None,
                            teacher_rollouts=False,
                            frozen_bundles=["old"] * 10,
                        )
                    )
                )
            ),
            config=dict(seed=8842, overrides={"train.gamma": 0.999, "train.learning_rate": 0.0002}),
        )
        training = self.parent / "run/training.json"
        training.write_text(json.dumps(record))
        self.identity = dict(
            policy_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            state_sha256=hashlib.sha256(self.state.read_bytes()).hexdigest(),
            run_sha256=hashlib.sha256(training.read_bytes()).hexdigest(),
            environment_sha256=[],
        )
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.manifest = self.root / "manifest.json"
        self.data = dict(
            schema=1,
            agent_steps=self.steps,
            **self.identity,
            frozen_bundles=[dict(directory="parent/bundle", sha256=self.identity["policy_sha256"])] * 10,
        )
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_preserves_optimizer_and_extends_absolute_counter(self):
        _, _, run, audit = load_continuation(self.manifest, self.steps)
        self.assertTrue(run["initialize"]["restore_learner"])
        self.assertEqual(run["total_timesteps"], 536_870_912)
        self.assertEqual(audit["additional_steps"], 268_435_456)
        self.assertEqual(run["overrides"]["train.learning_rate"], 0.0002)

    def test_corrupted_optimizer_is_rejected(self):
        self.state.write_bytes(self.state.read_bytes()[:-1] + b"\xff")
        with self.assertRaisesRegex(ValueError, "state_sha256"):
            load_continuation(self.manifest, self.steps)

    def test_wrong_training_identity_is_rejected(self):
        (self.parent / "run/training.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "run_sha256"):
            load_continuation(self.manifest, self.steps)

    def test_untrusted_pool_path_is_rejected(self):
        self.data["frozen_bundles"][0]["directory"] = "../outside"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_continuation(self.manifest, self.steps)

    def test_verified_but_nonfinite_optimizer_is_rejected(self):
        self.state.write_bytes(self.state.read_bytes()[:-4] + struct.pack("<f", float("nan")))
        self.data["state_sha256"] = hashlib.sha256(self.state.read_bytes()).hexdigest()
        self.identity["state_sha256"] = self.data["state_sha256"]
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "invalid optimizer"):
            load_continuation(self.manifest, self.steps)

    def test_invalid_new_step_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            load_continuation(self.manifest, 0)

    def test_evaluation_uses_resumed_counters_and_new_maps(self):
        with (
            patch.dict(os.environ, GENERALS_PILOT_CONTINUATION_MANIFEST=str(self.manifest)),
            patch.object(pilot, "STEPS", self.steps),
            patch.object(pilot, "export_and_audit") as export,
            patch.object(pilot, "command") as command,
        ):
            pilot.evaluate()
        self.assertEqual([x.args[0] for x in export.call_args_list], [536_870_912, 402_653_184])
        panels = [x for x in command.call_args_list if x.args[0] == "evaluate_spatial_population"]
        self.assertEqual(len(panels), 3)
        for call in panels:
            self.assertEqual(call.args[call.args.index("--seed") + 1], 37872)
            self.assertEqual(call.args[call.args.index("--sample-seed") + 1], 8754)


if __name__ == "__main__":
    unittest.main()

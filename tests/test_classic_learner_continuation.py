import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
                            opponent_weights=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 6],
                        )
                    )
                )
            ),
            config=dict(seed=8842, overrides={"train.gamma": 0.999, "train.learning_rate": 0.0002, "vec.total_agents": 8192, "train.horizon": 256}),
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
        frozen = []
        for i in range(10):
            directory = self.root / "frozen" / str(i)
            directory.mkdir(parents=True)
            policy = directory / "policy.bin"
            policy.write_bytes(struct.pack("<4f", i + 10, 0, 0, 0))
            frozen.append(dict(directory="frozen/" + str(i), sha256=hashlib.sha256(policy.read_bytes()).hexdigest()))
        self.manifest = self.root / "manifest.json"
        self.data = dict(
            schema=1,
            agent_steps=self.steps,
            **self.identity,
            frozen_bundles=frozen,
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

    def test_context_extension_is_bounded_and_preserves_parent(self):
        training = self.parent / "run/training.json"
        record = json.loads(training.read_text())
        record["build"]["config"]["fabric"] = dict(options=dict(context_radius=1.01))
        training.write_text(json.dumps(record))
        original = training.read_bytes()
        self.data["run_sha256"] = self.identity["run_sha256"] = hashlib.sha256(original).hexdigest()
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.data["context_extension"] = "zero_extend_radius2"
        self.write_manifest()
        _, build, run, audit = load_continuation(self.manifest, 33_554_432)
        self.assertEqual(build["fabric"]["options"]["context_radius"], 2.01)
        self.assertEqual(audit["context_extension"], "zero_extend_radius2")
        self.assertTrue(run["initialize"]["restore_learner"])
        self.assertEqual(training.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "isolated bounded"):
            load_continuation(self.manifest, 268_435_456)
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "isolated bounded"):
            load_continuation(self.manifest, 33_554_432)

    def test_continuation_accepts_completed_smaller_rollout_clock(self):
        training = self.parent / "run/training.json"
        record = json.loads(training.read_text())
        record["config"]["overrides"]["vec.total_agents"] = 2048
        training.write_text(json.dumps(record))
        state = bytearray(self.state.read_bytes())
        struct.pack_into("<Q", state, 8, self.steps // (2048 * 256))
        self.state.write_bytes(state)
        for key, path in (("run_sha256", training), ("state_sha256", self.state)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.identity[key] = self.data[key] = digest
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.write_manifest()
        _, _, run, receipt = load_continuation(self.manifest, 8_388_608)
        self.assertEqual(receipt["starting_agent_steps"], self.steps)
        self.assertEqual(run["overrides"]["vec.total_agents"], 2048)
        struct.pack_into("<Q", state, 8, self.steps // (8192 * 256))
        self.state.write_bytes(state)
        digest = hashlib.sha256(state).hexdigest()
        self.identity["state_sha256"] = self.data["state_sha256"] = digest
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "counters"):
            load_continuation(self.manifest, 8_388_608)

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

    def test_iterated_pool_retains_nine_and_adds_verified_parent(self):
        frozen = self.data["frozen_bundles"]
        self.data["frozen_bundles"] = frozen
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        original_training = (self.parent / "run/training.json").read_bytes()
        _, build, run, audit = load_continuation(self.manifest, self.steps)
        options = build["python_environment"]["options"]
        self.assertEqual(len(options["frozen_bundles"]), 10)
        self.assertEqual(options["frozen_bundles"][-1], str(self.parent / "bundle"))
        self.assertEqual(options["opponent_weights"], [1, 1, 1, 1, 1, 2, 2, 2, 2, 8, 8, 6])
        self.assertEqual(
            audit["frozen_policy_sha256"], [v["sha256"] for v in frozen[1:]] + [self.identity["policy_sha256"]]
        )
        self.assertEqual(audit["dropped_opponent_sha256"], frozen[0]["sha256"])
        self.assertTrue(run["initialize"]["restore_learner"])
        self.assertEqual((self.parent / "run/training.json").read_bytes(), original_training)

    def test_preserved_pool_rejects_duplicate_checkpoint(self):
        self.data["frozen_bundles"][1] = self.data["frozen_bundles"][0]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "distinct checkpoints"):
            load_continuation(self.manifest, self.steps)

    def test_iterated_pool_rejects_duplicate_parent(self):
        self.data["frozen_bundles"][0] = dict(directory="parent/bundle", sha256=self.identity["policy_sha256"])
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "already present"):
            load_continuation(self.manifest, self.steps)

    def test_unknown_pool_recipe_is_rejected(self):
        self.data["opponent_generation"] = "replace_everything"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unknown opponent generation"):
            load_continuation(self.manifest, self.steps)

    def test_native_siege_recipe_keeps_all_existing_opponents(self):
        self.data["scripted_opponent_generation"] = "append_classic_siege_padded_weight6"
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        _, build, run, audit = load_continuation(self.manifest, 8_388_608)
        options = build["python_environment"]["options"]
        self.assertEqual(options["scripted_opponents"],
                         ["expander_harvester", "sentinel", "classic_siege_padded"])
        self.assertEqual(options["opponent_weights"][-3:], [8, 6, 6])
        self.assertEqual(len(options["opponent_weights"]), 13)
        self.assertEqual(len(audit["frozen_policy_sha256"]), 10)
        self.assertEqual(audit["scripted_opponents"], options["scripted_opponents"])
        self.assertTrue(run["initialize"]["restore_learner"])

    def test_native_siege_addition_requires_bounded_qualification(self):
        self.data["scripted_opponent_generation"] = "append_classic_siege_padded_weight6"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "bounded 8M"):
            load_continuation(self.manifest, self.steps)

    def test_unknown_scripted_recipe_is_rejected(self):
        self.data["scripted_opponent_generation"] = "unreviewed_opponent"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unknown scripted"):
            load_continuation(self.manifest, self.steps)

    def native_parent(self, workers=1):
        training = self.parent / "run/training.json"
        record = json.loads(training.read_text())
        options = record["build"]["config"]["python_environment"]["options"]
        options["scripted_opponents"] = ["expander_harvester", "sentinel", "classic_siege_padded"]
        options["opponent_weights"].append(6)
        options["classic_siege_workers"] = workers
        training.write_text(json.dumps(record))
        self.identity["run_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
        self.data["run_sha256"] = self.identity["run_sha256"]
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.write_manifest()

    def test_worker_probe_preserves_pool_optimizer_and_parent_artifacts(self):
        self.native_parent()
        self.data["native_opponent_execution"] = "classic_siege_workers4"
        self.write_manifest()
        training = self.parent / "run/training.json"
        original = training.read_bytes()
        _, build, run, audit = load_continuation(self.manifest, 8_388_608)
        options = build["python_environment"]["options"]
        self.assertEqual(options["classic_siege_workers"], 4)
        self.assertEqual(audit["classic_siege_workers"], 4)
        self.assertEqual(audit["frozen_policy_sha256"], [e["sha256"] for e in self.data["frozen_bundles"]])
        original_options = json.loads(original)["build"]["config"]["python_environment"]["options"]
        self.assertEqual(options["opponent_weights"], original_options["opponent_weights"])
        self.assertTrue(run["initialize"]["restore_learner"])
        self.assertEqual(training.read_bytes(), original)

    def test_worker_probe_rejects_long_budget_or_simultaneous_pool_change(self):
        self.native_parent()
        self.data["native_opponent_execution"] = "classic_siege_workers4"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "isolated bounded 8M"):
            load_continuation(self.manifest, self.steps)
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "isolated bounded 8M"):
            load_continuation(self.manifest, 8_388_608)

    def test_worker_probe_requires_existing_native_siege(self):
        self.data["native_opponent_execution"] = "classic_siege_workers4"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "isolated bounded 8M"):
            load_continuation(self.manifest, 8_388_608)

    def test_default_preserves_previously_bound_workers(self):
        self.native_parent(workers=4)
        _, build, _, audit = load_continuation(self.manifest, self.steps)
        self.assertEqual(build["python_environment"]["options"]["classic_siege_workers"], 4)
        self.assertEqual(audit["native_opponent_execution"], "preserve")

    def reweight_panel(self):
        self.native_parent(workers=4)
        options = json.loads((self.parent / "run/training.json").read_text())["build"]["config"][
            "python_environment"]["options"]
        hashes = [e["sha256"] for e in self.data["frozen_bundles"]]
        names = ["frozen_" + sha[:12] for sha in hashes] + options["scripted_opponents"]
        panel = dict(checkpoint_sha256=self.identity["policy_sha256"],
                     frozen_policy_sha256=hashes, opponent_weights=options["opponent_weights"],
                     coworld_classic_rules=True, games=4096, wins=0, losses=0, draws=0,
                     by_opponent_and_seat={})
        for index, name in enumerate(names):
            seats = panel["by_opponent_and_seat"][name] = {}
            for side in (0, 1):
                games = 157 + (14 if index == side == 0 else 0)
                wins = games // 2 if name == "classic_siege_padded" else games * 9 // 10
                seats[str(side)] = dict(games=games, wins=wins, losses=games - wins, draws=0)
                panel["wins"] += wins
                panel["losses"] += games - wins
        self.data["opponent_weight_generation"] = "squared_nonwin_v1"
        self.write_panel(panel)
        return panel

    def write_panel(self, panel):
        path = self.root / "weight-panel.json"
        path.write_text(json.dumps(panel))
        self.data["opponent_evaluation"] = dict(
            file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self.write_manifest()

    def test_reweighting_retains_every_opponent_and_optimizer(self):
        self.reweight_panel()
        original = (self.parent / "run/training.json").read_bytes()
        _, build, run, audit = load_continuation(self.manifest, 33_554_432)
        weights = build["python_environment"]["options"]["opponent_weights"]
        self.assertEqual(len(weights), 13)
        self.assertTrue(all(1 <= value <= 24 for value in weights))
        self.assertGreater(weights[-1], 6)
        self.assertTrue(run["initialize"]["restore_learner"])
        self.assertEqual(run["total_timesteps"], self.steps + 33_554_432)
        self.assertEqual(audit["opponent_weight_audit"]["weights"], weights)
        self.assertEqual(audit["opponent_weight_audit"]["games"], 4096)
        self.assertEqual(audit["frozen_policy_sha256"], [x["sha256"] for x in self.data["frozen_bundles"]])
        self.assertEqual((self.parent / "run/training.json").read_bytes(), original)

    def test_reweighting_requires_isolated_short_pilot(self):
        self.reweight_panel()
        with self.assertRaisesRegex(ValueError, "isolated bounded 33M"):
            load_continuation(self.manifest, self.steps)
        self.data["opponent_generation"] = "drop_oldest_append_parent_weight8"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "isolated bounded 33M"):
            load_continuation(self.manifest, 33_554_432)

    def test_reweighting_rejects_stale_policy_and_bad_counts(self):
        panel = self.reweight_panel()
        panel["checkpoint_sha256"] = "wrong"
        self.write_panel(panel)
        with self.assertRaisesRegex(ValueError, "resumed policy and pool"):
            load_continuation(self.manifest, 33_554_432)
        panel["checkpoint_sha256"] = self.identity["policy_sha256"]
        panel["by_opponent_and_seat"]["classic_siege_padded"]["0"]["wins"] += 1
        self.write_panel(panel)
        with self.assertRaisesRegex(ValueError, "inconsistent outcome counts"):
            load_continuation(self.manifest, 33_554_432)

    def test_reweighting_rejects_untrusted_or_corrupted_evaluation(self):
        self.reweight_panel()
        self.data["opponent_evaluation"]["file"] = "../outside.json"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_continuation(self.manifest, 33_554_432)
        self.data["opponent_evaluation"]["file"] = "weight-panel.json"
        self.write_manifest()
        (self.root / "weight-panel.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_continuation(self.manifest, 33_554_432)

    def test_reweighting_requires_explicit_recipe(self):
        self.reweight_panel()
        self.data["opponent_weight_generation"] = "preserve"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "explicit reweighting"):
            load_continuation(self.manifest, 33_554_432)

    def test_preserves_previously_reweighted_native_pool(self):
        self.native_parent(workers=4)
        training = self.parent / "run/training.json"
        record = json.loads(training.read_text())
        weights = [1, 2, 1, 1, 1, 4, 5, 5, 11, 17, 3, 4, 16]
        record["build"]["config"]["python_environment"]["options"]["opponent_weights"] = weights
        training.write_text(json.dumps(record))
        self.identity["run_sha256"] = hashlib.sha256(training.read_bytes()).hexdigest()
        self.data["run_sha256"] = self.identity["run_sha256"]
        Path(str(self.state) + ".json").write_text(json.dumps(self.identity))
        self.write_manifest()
        _, build, _, audit = load_continuation(self.manifest, self.steps)
        self.assertEqual(build["python_environment"]["options"]["opponent_weights"], weights)
        self.assertEqual(audit["opponent_weight_generation"], "preserve")

    def test_unknown_worker_recipe_is_rejected(self):
        self.data["native_opponent_execution"] = "all_cpus"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unknown native opponent"):
            load_continuation(self.manifest, 8_388_608)

    def test_invalid_inherited_worker_count_is_rejected(self):
        self.native_parent(workers=True)
        with self.assertRaisesRegex(ValueError, "Invalid inherited"):
            load_continuation(self.manifest, 8_388_608)




if __name__ == "__main__":
    unittest.main()

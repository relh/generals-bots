"""Preflight must reject plausible configurations that would misqualify a run."""

import copy
import unittest

from integrations.classic_contract import ENGINE_SHA256, validate_training_contract, verify_engine


class ClassicContractTests(unittest.TestCase):
    def setUp(self):
        self.build = {
            "python_environment": {
                "options": {
                    "coworld_classic": True,
                    "horizon": 2000,
                    "shaping_gamma": 0.999,
                    "parallel_games": 2048,
                    "balance_opponent_sides": True,
                },
                "spec": {"agents": 2048, "observation_size": 7056, "action_sizes": [3529]},
            },
            "fabric": {"observation_size": 7056, "action_sizes": [3529]},
        }
        self.run = {
            "overrides": {
                "train.gamma": 0.999,
                "vec.total_agents": 2048,
                "train.horizon": 256,
                "train.minibatch_size": 8192,
            }
        }

    def test_pinned_engine_and_actual_rollout(self):
        self.assertEqual(verify_engine(), ENGINE_SHA256)
        result = validate_training_contract(self.build, self.run)
        self.assertEqual(result["training_geometry"]["steps_per_epoch"], 524288)

    def test_drift_rejected(self):
        for key, value in [
            ("coworld_classic", False),
            ("coworld_small_map_curriculum", True),
            ("horizon", 1999),
            ("shaping_gamma", 0.99),
            ("parallel_games", 8192),
            ("balance_opponent_sides", False),
        ]:
            with self.subTest(key=key):
                build = copy.deepcopy(self.build)
                build["python_environment"]["options"][key] = value
                with self.assertRaises(ValueError):
                    validate_training_contract(build, self.run)

    def test_invalid_discount_or_partial_minibatch_rejected(self):
        for key, value in [
            ("train.gamma", float("nan")),
            ("train.gamma", True),
            ("train.minibatch_size", 8193),
            ("vec.total_agents", 8192),
        ]:
            with self.subTest(key=key):
                run = copy.deepcopy(self.run)
                run["overrides"][key] = value
                with self.assertRaises(ValueError):
                    validate_training_contract(self.build, run)

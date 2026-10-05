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


def test_projection_validates_fixed_source_fields_and_does_not_mutate_source():
    from integrations.classic_contract import project_current_options

    source = {
        "parallel_games": 8192,
        "coworld_classic": True,
        "directional_features": True,
        "public_scalar_features": True,
        "teacher": None,
        "imitation_weight": 0.0,
        "frozen_bundles": ["source-one", "source-two"],
    }
    projected = project_current_options(source)
    assert set(projected) == {"parallel_games", "frozen_bundles"}
    projected["frozen_bundles"].append("another")
    assert source["frozen_bundles"] == ["source-one", "source-two"]
    assert project_current_options({"terminal_reward_mode": "signed", "horizon": 4}) == {
        "terminal_reward_mode": "signed",
        "horizon": 4,
    }


def test_projection_rejects_unsupported_effective_features_and_unknown_fields():
    import pytest

    from integrations.classic_contract import project_current_options

    for unsupported in (
        {"teacher": "sentinel"},
        {"coworld_classic": False},
        {"public_scalar_ablation": True},
        {"land_gain_reward_weight": 0.2},
        {"unrecognized_feature": False},
    ):
        with pytest.raises(ValueError):
            project_current_options(unsupported)

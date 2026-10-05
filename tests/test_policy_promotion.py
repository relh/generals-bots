import copy
import unittest

from integrations.policy_promotion import promotion_report, wilson_lower


def hosted(games=128, wins=90):
    return {
        "completed": games * 4,
        "failed": 0,
        "by_opponent_and_seat": {
            f"{name}-seat{seat}": {"games": games, "wins": wins, "losses": games - wins}
            for name in ("leader", "relh")
            for seat in (0, 1)
        },
    }


class PromotionTests(unittest.TestCase):
    def test_actual_hosted_baseline_is_not_ready(self):
        report = promotion_report(
            {
                "completed": 64,
                "failed": 0,
                "by_opponent_and_seat": {
                    "leader-seat0": {"games": 16, "wins": 3, "losses": 13},
                    "leader-seat1": {"games": 16, "wins": 6, "losses": 10},
                    "relh-seat0": {"games": 16, "wins": 9, "losses": 7},
                    "relh-seat1": {"games": 16, "wins": 9, "losses": 7},
                },
            },
            ["leader", "relh"],
        )
        self.assertFalse(report["strength_gate_passes"])
        self.assertEqual(report["opponents"]["leader"]["win_rate"], 9 / 32)

    def test_enough_balanced_wins_pass(self):
        self.assertTrue(promotion_report(hosted(), ["leader", "relh"])["strength_gate_passes"])

    def test_missing_required_opponent_and_unbalanced_seats_fail(self):
        data = hosted()
        data["by_opponent_and_seat"]["leader-seat0"].update(games=129, wins=91)
        data["completed"] += 1
        report = promotion_report(data, ["leader", "relh", "third"])
        self.assertFalse(report["strength_gate_passes"])
        self.assertIn("missing or unbalanced player seats", report["opponents"]["leader"]["reasons"])
        self.assertEqual(report["opponents"]["third"]["games"], 0)

    def test_small_perfect_screen_and_failed_request_fail(self):
        self.assertFalse(promotion_report(hosted(16, 16), ["leader", "relh"])["strength_gate_passes"])
        data = hosted()
        data["failed"] = 1
        self.assertFalse(promotion_report(data, ["leader", "relh"])["strength_gate_passes"])

    def test_population_schema_counts_draws_as_nonwins(self):
        data = {
            "games": 256,
            "by_opponent_and_seat": {
                "siege": {str(side): {"games": 128, "wins": 84, "losses": 40, "draws": 4} for side in (0, 1)}
            },
        }
        report = promotion_report(data, ["siege"])
        self.assertTrue(report["strength_gate_passes"])
        self.assertEqual(report["opponents"]["siege"]["win_rate"], 168 / 256)

    def test_invalid_counts_and_inconsistent_totals_rejected(self):
        for mutation in (
            lambda d: d.update(completed=511),
            lambda d: d["by_opponent_and_seat"]["leader-seat0"].update(wins=129),
            lambda d: d.update(failed=True),
        ):
            data = copy.deepcopy(hosted())
            mutation(data)
            with self.assertRaises(ValueError):
                promotion_report(data, ["leader", "relh"])

    def test_confidence_requirement_is_independent_of_point_target(self):
        self.assertLess(wilson_lower(13, 20), 0.5)
        self.assertGreater(wilson_lower(168, 256), 0.5)
        report = promotion_report(hosted(10, 7), ["leader", "relh"], min_games=2)
        self.assertFalse(report["strength_gate_passes"])

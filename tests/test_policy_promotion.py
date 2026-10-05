import copy
import unittest

from integrations.policy_promotion import strength_report, wilson_lower


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
        report = strength_report(
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
        self.assertTrue(strength_report(hosted(), ["leader", "relh"])["strength_gate_passes"])

    def test_missing_required_opponent_and_unbalanced_seats_fail(self):
        data = hosted()
        data["by_opponent_and_seat"]["leader-seat0"].update(games=129, wins=91)
        data["completed"] += 1
        report = strength_report(data, ["leader", "relh", "third"])
        self.assertFalse(report["strength_gate_passes"])
        self.assertIn("missing or unbalanced player seats", report["opponents"]["leader"]["reasons"])
        self.assertEqual(report["opponents"]["third"]["games"], 0)

    def test_small_perfect_screen_and_failed_request_fail(self):
        self.assertFalse(strength_report(hosted(16, 16), ["leader", "relh"])["strength_gate_passes"])
        data = hosted()
        data["failed"] = 1
        self.assertFalse(strength_report(data, ["leader", "relh"])["strength_gate_passes"])

    def test_population_schema_counts_draws_as_nonwins(self):
        data = {
            "games": 256,
            "by_opponent_and_seat": {
                "siege": {str(side): {"games": 128, "wins": 84, "losses": 40, "draws": 4} for side in (0, 1)}
            },
        }
        report = strength_report(data, ["siege"])
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
                strength_report(data, ["leader", "relh"])

    def test_confidence_requirement_is_independent_of_point_target(self):
        self.assertLess(wilson_lower(13, 20), 0.5)
        self.assertGreater(wilson_lower(168, 256), 0.5)
        report = strength_report(hosted(10, 7), ["leader", "relh"], min_games=2)
        self.assertFalse(report["strength_gate_passes"])


def prepared_panel(directory, *, pending=False):
    from integrations.hosted_policy import load_panel, prepare, save, summarize

    policy = '5ef78e23-c02e-4d13-bc8e-2f9ab47fb0a1'
    intent = prepare(policy, {
        'incumbent': 'e53e30be-0b23-4d62-b944-4dd249a483fe',
        'Daveey': '76b0a083-f0a4-4ec7-9811-038349266633',
    }, 512, 'confirmation', 'f' * 64, 'bd9bb9d', 'sha256:' + 'a' * 64)
    directory.mkdir()
    save(directory / 'intent.json', intent)
    states = {}
    for label, body in intent['requests'].items():
        save(directory / (label + '-payload.json'), body)
        request_id = 'xreq_' + label
        save(directory / (label + '-receipt.json'), {'id': request_id})
        seat = body['roster'][0]['slot']
        completed = body['num_episodes'] - int(pending)
        states[label] = {
            'id': request_id, 'status': 'running' if pending else 'completed',
            'completed_count': completed, 'failed_count': 0,
            'episodes': [{
                'id': f'ereq_{label}_{i}', 'episode_id': f'episode_{label}_{i}',
                'status': 'completed', 'cost_usd': 0.001,
                'participants': [
                    {'position': seat, 'policy_version_id': policy},
                    {'position': 1 - seat, 'policy_version_id': body['roster'][1]['player']['policy_ref']},
                ],
                'scores': [{'policy_version_id': policy, 'score': 1}],
            } for i in range(completed)],
        }
        save(directory / (label + '-state.json'), states[label])
    return summarize(load_panel(directory), states)


def operational_report(summary, directory, **identity):
    from integrations.policy_promotion import promotion_report
    return promotion_report(summary, ['incumbent', 'Daveey'], panel=directory,
                            checkpoint_sha256=identity.get('checkpoint_sha256', 'f' * 64),
                            image_digest=identity.get('image_digest', 'sha256:' + 'a' * 64),
                            source_commit=identity.get('source_commit', 'bd9bb9d'))


def test_operational_promotion_requires_complete_current_panel(tmp_path):
    directory = tmp_path / 'complete'
    summary = prepared_panel(directory)
    assert operational_report(summary, directory)['strength_gate_passes']
    directory = tmp_path / 'pending'
    summary = prepared_panel(directory, pending=True)
    assert summary['completed'] == 1020 and summary['pending'] == 4
    counts_only = dict(summary, pending=0)
    assert strength_report(counts_only, ['incumbent', 'Daveey'])['strength_gate_passes']
    report = operational_report(summary, directory)
    assert not report['strength_gate_passes'] and not report['evidence_gate_passes']
    assert report['pending_episodes'] == 4


def test_operational_promotion_rejects_provenance_and_receipt_tampering(tmp_path):
    import json

    import pytest

    from integrations.hosted_policy import save

    directory = tmp_path / 'panel'
    summary = prepared_panel(directory)
    for identity in ({'checkpoint_sha256': 'b' * 64}, {'source_commit': '1234567'},
                     {'image_digest': 'sha256:' + 'c' * 64}):
        with pytest.raises(ValueError, match='identity differs'):
            operational_report(summary, directory, **identity)
    with pytest.raises(ValueError, match='current hosted results schema'):
        operational_report(dict(summary, schema='older-schema'), directory)
    altered = dict(summary, cost_usd=99.0)
    with pytest.raises(ValueError, match='differs from retained panel states'):
        operational_report(altered, directory)
    path = directory / 'incumbent-seat0-receipt.json'
    receipt = json.loads(path.read_text())
    save(path, {'id': receipt['id'] + '-different'})
    with pytest.raises(ValueError, match='receipt identity'):
        operational_report(summary, directory)

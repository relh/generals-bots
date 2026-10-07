import json

import pytest

from integrations.hosted_policy import load_panel, prepare, save, status, submit, summarize
from integrations.policy_promotion import strength_report

POLICY = '5ef78e23-c02e-4d13-bc8e-2f9ab47fb0a1'
OPPONENT = 'e53e30be-0b23-4d62-b944-4dd249a483fe'


def panel(games=2):
    return prepare(POLICY, {'incumbent': OPPONENT}, games, 'test-native-smoke',
                   'f' * 64, '57be9f9', 'sha256:' + 'a' * 64)


@pytest.mark.parametrize('games', [0, 1, 3, -2, True, 4098])
def test_panel_rejects_unbalanced_or_unbounded_budgets(games):
    with pytest.raises(ValueError):
        panel(games)


class Client:
    def __init__(self):
        self.created = []

    def request(self, path, body=None):
        if body:
            self.created.append(body)
            return {'id': f'xreq_{len(self.created)}'}
        if path.startswith('/v2/experience-requests?'):
            return {'entries': [], 'next_cursor': None}
        return {'id': path.rsplit('/', 1)[1]}


def test_preserved_panel_resumes_without_extra_games_and_rejects_tampering(tmp_path):
    directory = tmp_path / 'panel'
    client = Client()
    submit(directory, panel(), client)
    assert [b['num_episodes'] for b in client.created] == [1, 1]
    assert [b['roster'][0]['slot'] for b in client.created] == [0, 1]
    submit(directory, panel(), client)
    assert len(client.created) == 2
    body = directory / 'incumbent-seat0-payload.json'
    changed = json.loads(body.read_text())
    changed['num_episodes'] = 99
    save(body, changed)
    with pytest.raises(ValueError, match='payload changed'):
        load_panel(directory)


def states(intent):
    result = {}
    for label, body in intent['requests'].items():
        seat = body['roster'][0]['slot']
        result[label] = {
            'id': 'xreq_' + label, 'status': 'completed', 'completed_count': 1, 'failed_count': 0,
            'episodes': [{
                'id': 'ereq_' + label, 'episode_id': 'episode_' + label,
                'status': 'completed', 'cost_usd': 0.002,
                'participants': [{'position': seat, 'policy_version_id': POLICY},
                                 {'position': 1 - seat, 'policy_version_id': OPPONENT}],
                'scores': [{'policy_version_id': POLICY, 'score': 1}],
            }],
        }
    return result


def test_frozen_balanced_results_integrate_with_promotion_and_preserve_failures():
    intent = panel()
    current = states(intent)
    summary = summarize(intent, current)
    assert summary['completed'] == 2 and summary['cost_usd'] == 0.004
    report = strength_report(summary, ['incumbent'], min_games=2)
    assert report['failed_requests'] == 0
    assert not report['strength_gate_passes']  # Two wins cannot establish confidence.
    current['incumbent-seat0']['status'] = 'failed'
    failed = summarize(intent, current)
    assert failed['failed'] == 1
    assert not strength_report(failed, ['incumbent'], min_games=2)['strength_gate_passes']


def test_collection_rejects_changed_policy_identity():
    intent = panel()
    current = states(intent)
    current['incumbent-seat0']['episodes'][0]['participants'][0]['policy_version_id'] = OPPONENT
    with pytest.raises(ValueError, match='Frozen episode roster'):
        summarize(intent, current)


@pytest.mark.parametrize('accepted_before_limit', [0, 1])
def test_partial_submission_polls_only_receipts_and_resumes_same_intent(tmp_path, accepted_before_limit):
    intent = panel()
    completed = states(intent)

    class LimitedClient(Client):
        limit = accepted_before_limit

        def request(self, path, body=None):
            if body:
                if len(self.created) >= self.limit:
                    raise RuntimeError('Observatory HTTP 429')
                self.created.append(body)
                label = next(name for name, payload in intent['requests'].items() if payload == body)
                return {'id': completed[label]['id']}
            if '?' in path:
                return {'entries': [], 'next_cursor': None}
            return next(value for value in completed.values() if value['id'] == path.rsplit('/', 1)[1])

    directory = tmp_path / 'partial'
    client = LimitedClient()
    with pytest.raises(RuntimeError, match='429'):
        submit(directory, intent, client)
    observed = status(directory, client)
    summary = summarize(intent, observed)
    assert summary['completed'] == accepted_before_limit
    assert summary['pending'] == 2 - accepted_before_limit
    assert len(summary['unsubmitted_requests']) == 2 - accepted_before_limit
    assert len(list(directory.glob('*-receipt.json'))) == accepted_before_limit
    assert len(list(directory.glob('*-state.json'))) == accepted_before_limit
    assert not strength_report(summary, ['incumbent'], min_games=2)['strength_gate_passes']
    client.limit = 2
    submit(directory, intent, client)
    assert client.created == list(intent['requests'].values())
    final = summarize(intent, status(directory, client))
    assert final['completed'] == 2 and final['pending'] == 0
    assert final['unsubmitted_requests'] == {}


def test_large_panel_chunks_requests_and_aggregates_each_seat():
    intent = panel(256)
    assert len(intent['requests']) == 4
    assert [body['num_episodes'] for body in intent['requests'].values()] == [100, 28, 100, 28]
    current = states(intent)
    for label, state in current.items():
        body = intent['requests'][label]
        first = state['episodes'][0]
        state['episodes'] = [
            {**first, 'id': f'ereq_{label}_{i}', 'episode_id': f'episode_{label}_{i}'}
            for i in range(body['num_episodes'])
        ]
        state['completed_count'] = body['num_episodes']
    result = summarize(intent, current)
    assert result['completed'] == 256
    assert result['by_opponent_and_seat'] == {
        'incumbent-seat0': {'games': 128, 'wins': 128, 'losses': 0, 'draws': 0},
        'incumbent-seat1': {'games': 128, 'wins': 128, 'losses': 0, 'draws': 0},
    }
    assert strength_report(result, ['incumbent'], min_games=256)['strength_gate_passes']

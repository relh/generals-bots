import copy

import pytest

from integrations.audit_hosted_panel_replays import audit_replay


def test_audit_replay_checks_candidate_move_against_previous_public_frame():
    frame = {'turn': 0, 'type_grid': [[1] * 18 for _ in range(18)],
             'owner_grid': [[0] * 18 for _ in range(18)],
             'army_grid': [[0] * 18 for _ in range(18)]}
    frame['owner_grid'][4][5] = 2
    frame['army_grid'][4][5] = 3
    replay = {'ruleset': 'classic', 'seed': 123, 'height': 18, 'width': 18,
              'frames': [frame, {**frame, 'turn': 1}],
              'turns': [{'turn': 0, 'actions': [[1, 0, 0, 0, 0], [0, 4, 5, 3, 0]],
                         'timed_out': [False, False], 'forfeited': [], 'applied': True}],
              'result': {'turns': 1, 'scores': [-1, 1], 'timeouts': [0, 0],
                         'reason': 'general_capture'}}
    episode = {'episode_id': 'episode', 'opponent_seat': 'Daveey-seat1', 'score': 1}
    assert audit_replay(replay, episode)['seat'] == 1

    illegal = copy.deepcopy(replay)
    illegal['frames'][0]['army_grid'][4][5] = 1
    with pytest.raises(ValueError, match='Illegal candidate move'):
        audit_replay(illegal, episode)

    timeout = copy.deepcopy(replay)
    timeout['turns'][0]['timed_out'][1] = True
    with pytest.raises(ValueError, match='timed out or forfeited'):
        audit_replay(timeout, episode)

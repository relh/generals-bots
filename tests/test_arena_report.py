from generals.evaluation.report import matchup_report


def rows(maps, results):
    return [dict(board_id=str(board), result=result, own_invalid_moves='0', own_builds='0')
            for board in range(maps) for result in results]


def test_gate_requires_maps_and_counts_draws_as_nonwins():
    assert not matchup_report(rows(8, ['win'] * 4))['gate']
    assert matchup_report(rows(64, ['win'] * 4))['gate']
    report = matchup_report(rows(64, ['win', 'win', 'win', 'draw']))
    assert report['win_rate'] == .75
    assert not report['gate']


def test_interval_resamples_whole_maps():
    data = rows(32, ['win'] * 4)
    data += [dict(row, board_id=str(int(row['board_id']) + 32), result='loss') for row in data]
    report = matchup_report(data)
    assert report['win_rate'] == .5
    assert report['win_ci95'][0] < .4
    assert report['win_ci95'][1] > .6

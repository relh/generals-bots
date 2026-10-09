"""Download and audit every replay in a terminal hosted Classic panel.

The summary and replay bytes stay under the panel directory. Signed replay URLs
are used in memory only; the output records episode IDs and SHA-256 hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from uuid import UUID

from integrations.hosted_policy import Observatory, save

DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def audit_replay(replay, episode):
    """Check public Classic receipts and the candidate's recorded action."""
    if replay.get('ruleset') != 'classic':
        raise ValueError('Replay is not Classic')
    seat_label = episode['opponent_seat']
    if not seat_label.endswith(('seat0', 'seat1')):
        raise ValueError('Summary has no candidate seat')
    seat = int(seat_label[-1])
    turns, frames, result = replay['turns'], replay['frames'], replay['result']
    if not isinstance(replay['seed'], int) or isinstance(replay['seed'], bool):
        raise ValueError('Replay seed is not an integer')
    if len(frames) != len(turns) + 1 or result['turns'] != len(turns):
        raise ValueError('Replay frames and turn count disagree')
    if result['scores'][seat] != episode['score']:
        raise ValueError('Replay result differs from panel score')
    if result['timeouts'][seat] != 0:
        raise ValueError('Candidate has a replay timeout')
    height, width = replay['height'], replay['width']
    if not (18 <= height <= 21 and 18 <= width <= 21):
        raise ValueError('Replay map dimensions differ from hosted Classic')
    for index, receipt in enumerate(turns):
        frame = frames[index]
        if frame['turn'] != index or receipt['turn'] != index or frames[index + 1]['turn'] != index + 1:
            raise ValueError(f'Turn index differs at {index}')
        if not receipt['applied']:
            raise ValueError(f'Unapplied turn {index}')
        if receipt['timed_out'][seat] or seat in receipt['forfeited']:
            raise ValueError(f'Candidate timed out or forfeited at turn {index}')
        action = receipt['actions'][seat]
        if len(action) != 5 or any(type(value) is not int for value in action):
            raise ValueError(f'Malformed candidate action at turn {index}')
        kind, row, col, direction, half = action
        if kind not in (0, 1) or not (0 <= row < height and 0 <= col < width):
            raise ValueError(f'Candidate action outside map at turn {index}')
        if direction not in range(4) or half not in (0, 1):
            raise ValueError(f'Candidate direction or split invalid at turn {index}')
        if kind == 1:
            continue
        next_row, next_col = row + DELTAS[direction][0], col + DELTAS[direction][1]
        if (frame['owner_grid'][row][col] != seat + 1
                or frame['army_grid'][row][col] < 2
                or not (0 <= next_row < height and 0 <= next_col < width)
                or frame['type_grid'][next_row][next_col] == 2):
            raise ValueError(f'Illegal candidate move at turn {index}')
    return {'episode_id': episode['episode_id'], 'seat': seat, 'score': episode['score'],
            'seed': replay['seed'], 'turns': len(turns), 'reason': result['reason'],
            'candidate_timeouts': 0, 'candidate_forfeits': 0,
            'unapplied_turns': 0, 'illegal_candidate_actions': 0}


def download(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ValueError('Replay URL must use HTTPS')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if not data:
                raise ValueError('Replay download is empty')
            return data
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, urllib.error.HTTPError)
                                and error.code < 500 and error.code != 429):
                raise RuntimeError('Replay download failed; retry audit after checking episode access') from None
            time.sleep(2 ** attempt)
    raise AssertionError('Unreachable')


def audit_panel(directory, client, workers=4):
    summary = json.loads((directory / 'summary.json').read_text())
    if summary['schema'] != 'generals-hosted-results-v1':
        raise ValueError('Unsupported panel summary')
    episodes = summary['episodes']
    if summary['pending'] or summary['failed'] or summary['completed'] != len(episodes):
        raise ValueError('Panel must be complete with zero failures before replay audit')
    if len({episode['episode_id'] for episode in episodes}) != len(episodes):
        raise ValueError('Duplicate panel episode ID')
    replay_dir = directory / 'replays'
    replay_dir.mkdir(exist_ok=True)
    prior_path = directory / 'replay-audit.json'
    prior = (json.loads(prior_path.read_text()) if prior_path.exists() else None)
    prior_hashes = ({item['episode_id']: item['replay_sha256'] for item in prior['episodes']}
                    if prior else {})

    def one(episode):
        episode_id = episode['episode_id']
        if str(UUID(episode_id)) != episode_id:
            raise ValueError('Invalid panel episode ID')
        path = replay_dir / f'{episode_id}.bin'
        if path.exists():
            data = path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    detail = client.request('/v2/episodes/' + episode_id)
                    break
                except RuntimeError:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            if detail.get('id') != episode_id or not detail.get('replay_url'):
                raise ValueError(f'No replay for episode {episode_id}')
            data = download(detail['replay_url'])
            # Keep the exact downloaded compressed artifact for repeatable checks.
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
        try:
            replay = json.loads(gzip.decompress(data))
        except (OSError, json.JSONDecodeError):
            raise ValueError(f'Invalid gzip JSON replay for episode {episode_id}') from None
        result = audit_replay(replay, episode)
        result['replay_sha256'] = hashlib.sha256(data).hexdigest()
        if episode_id in prior_hashes and prior_hashes[episode_id] != result['replay_sha256']:
            raise ValueError(f'Preserved replay checksum differs for episode {episode_id}')
        result['replay_bytes'] = len(data)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        audited = list(pool.map(one, episodes))
    seeds = [item['seed'] for item in audited]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate initial-state seed in hosted panel')
    report = {'schema': 'generals-hosted-replay-audit-v1',
              'policy_id': summary['policy_id'], 'completed': len(audited),
              'unique_seeds': len(seeds), 'total_turns': sum(item['turns'] for item in audited),
              'replay_bytes': sum(item['replay_bytes'] for item in audited),
              'episodes': audited}
    save(directory / 'replay-audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    report = audit_panel(args.panel, Observatory(), args.workers)
    print(json.dumps({key: report[key] for key in
                      ('completed', 'unique_seeds', 'total_turns', 'replay_bytes')}))


if __name__ == '__main__':
    main()

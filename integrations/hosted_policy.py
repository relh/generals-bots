"""Bounded hosted Classic panels using the current Observatory HTTP contract.

Prepare with submit --dry-run; submit the preserved intent, then status/collect.
Image build and registration use Docker and the installed Coworld SDK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from uuid import UUID

SERVER = 'https://softmax.com/api'
DIVISION = 'div_5ee4b276-f330-42e8-b8e4-a6097c779d99'


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def save(path, value):
    path.write_bytes(encode(value))


def prepare(policy, opponents, games, key, checkpoint, source, image):
    """One request per opponent and seat, with an explicit finite game budget."""
    UUID(policy)
    if not isinstance(games, int) or isinstance(games, bool) or games < 2 or games % 2:
        raise ValueError('games per opponent must be a positive even count of at least two')
    if not opponents or len(opponents) > 16 or games * len(opponents) > 4096:
        raise ValueError('Panel requires 1–16 opponents and at most 4096 total episodes')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,120}', key):
        raise ValueError('Panel key must contain 1–120 letters, digits, underscores or hyphens')
    if not re.fullmatch(r'[0-9a-f]{64}', checkpoint) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
        raise ValueError('Specify full checkpoint SHA-256 and immutable image digest')
    if not re.fullmatch(r'[0-9a-f]{7,40}', source):
        raise ValueError('Specify source Git commit')
    requests = {}
    for name, opponent in opponents.items():
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('Opponent names must be letters, digits, underscores or hyphens')
        UUID(opponent)
        if opponent == policy:
            raise ValueError('Candidate cannot be its own qualification opponent')
        for seat in (0, 1):
            label = f'{name}-seat{seat}'
            requests[label] = {
                'idempotency_key': f'{key}-{label}', 'private': True,
                'target': {'division_id': DIVISION, 'variant_id': 'competition'},
                'roster': [
                    {'player': {'policy_ref': policy}, 'slot': seat},
                    {'player': {'policy_ref': opponent}, 'slot': 1 - seat},
                ],
                'num_episodes': games // 2, 'execution_backend': 'k8s',
                'notes': f'Frozen Classic panel {key}; checkpoint {checkpoint}; source {source}; balanced seats.',
            }
    return {'schema': 'generals-hosted-panel-v1', 'policy_id': policy,
            'checkpoint_sha256': checkpoint, 'source_commit': source, 'image_digest': image,
            'opponents': opponents, 'games_per_opponent': games,
            'total_episodes': games * len(opponents), 'requests': requests,
            'payload_sha256': {name: digest(body) for name, body in requests.items()}}


class Observatory:
    def __init__(self, server=SERVER):
        self.server = server.rstrip('/')
        self.token = os.environ.get('SOFTMAX_TOKEN')
        if not self.token:
            try:
                from softmax.auth import load_user_token
                self.token = load_user_token(server=self.server)
            except Exception:  # noqa: BLE001 - redact credential-loader exception details
                raise RuntimeError('Cannot load canonical Softmax user credentials') from None
        if not self.token:
            raise ValueError('Use SOFTMAX_TOKEN or the canonical Softmax user login')
        if self.token.startswith('ply_'):
            raise ValueError('Hosted panels require a user token, not a player session')

    def request(self, path, body=None):
        request = urllib.request.Request(
            self.server + '/observatory' + path,
            data=None if body is None else encode(body),
            headers={'Authorization': 'Bearer ' + self.token,
                     'Content-Type': 'application/json', 'User-Agent': 'coworld-cli/0.1'},
            method='GET' if body is None else 'POST',
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Observatory HTTP {error.code}; preserve intent and receipts before retry') from None
        except urllib.error.URLError:
            raise RuntimeError('Observatory transport failed; reconcile preserved intent before retry') from None


def load_panel(directory):
    panel = json.loads((directory / 'intent.json').read_text())
    if panel['schema'] != 'generals-hosted-panel-v1':
        raise ValueError('Unsupported panel intent')
    for label, body in panel['requests'].items():
        expected = panel['payload_sha256'][label]
        actual = json.loads((directory / f'{label}-payload.json').read_text())
        if digest(body) != expected or digest(actual) != expected:
            raise ValueError(f'Preserved payload changed: {label}')
    return panel


def submit(directory, panel, client):
    if directory.exists():
        if load_panel(directory) != panel:
            raise ValueError('Output directory belongs to a different panel intent')
    else:
        directory.mkdir(parents=True)
        for label, body in panel['requests'].items():
            save(directory / f'{label}-payload.json', body)
        save(directory / 'intent.json', panel)
    # Read current cursor schema directly; no outdated typed SDK page parser.
    existing = client.request('/v2/experience-requests?mine=true&limit=100')
    if not isinstance(existing['entries'], list) or 'next_cursor' not in existing:
        raise ValueError('Unexpected current Experience Request list schema')
    save(directory / 'existing-before-submit.json', existing)
    for label, body in panel['requests'].items():
        receipt = directory / f'{label}-receipt.json'
        if receipt.exists():
            detail = client.request('/v2/experience-requests/' + json.loads(receipt.read_text())['id'])
            save(directory / f'{label}-state.json', detail)
            continue
        # Backend idempotency guarantees the same key cannot schedule extra games,
        # including a lost response after successful submission.
        result = client.request('/v2/experience-requests', body)
        save(receipt, result)
        print(json.dumps({'opponent_seat': label, 'request_id': result['id']}), flush=True)


def status(directory, client):
    panel = load_panel(directory)
    states = {}
    for label in panel['requests']:
        receipt = json.loads((directory / f'{label}-receipt.json').read_text())
        state = client.request('/v2/experience-requests/' + receipt['id'])
        if state['id'] != receipt['id']:
            raise ValueError('Request identity changed')
        save(directory / f'{label}-state.json', state)
        states[label] = state
    return states


def summarize(panel, states):
    summary = {'schema': 'generals-hosted-results-v1', 'policy_id': panel['policy_id'],
               'checkpoint_sha256': panel['checkpoint_sha256'], 'source_commit': panel['source_commit'],
               'image_digest': panel['image_digest'], 'opponents': panel['opponents'],
               'request_payload_sha256': panel['payload_sha256'],
               'completed': 0, 'failed': 0, 'pending': 0, 'cost_usd': 0.0,
               'request_ids': [], 'by_opponent_and_seat': {}, 'episodes': []}
    seen = set()
    for label, body in panel['requests'].items():
        state = states[label]
        summary['request_ids'].append(state['id'])
        counts = {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0}
        summary['by_opponent_and_seat'][label] = counts
        failures = state['failed_count']
        if state['status'] in ('failed', 'canceled', 'cancelled') and failures == 0:
            failures = 1
        summary['failed'] += failures
        summary['pending'] += max(0, body['num_episodes'] - state['completed_count'] - state['failed_count'])
        for episode in state['episodes']:
            if episode['status'] != 'completed':
                continue
            eid = episode['episode_id']
            if eid in seen:
                raise ValueError('Duplicate completed episode in panel')
            seen.add(eid)
            seat = body['roster'][0]['slot']
            participants = episode['participants']
            if {p['position']: p['policy_version_id'] for p in participants} != {
                seat: panel['policy_id'], 1 - seat: body['roster'][1]['player']['policy_ref'],
            }:
                raise ValueError('Frozen episode roster differs from preserved request')
            scores = [s['score'] for s in episode['scores'] if s['policy_version_id'] == panel['policy_id']]
            if len(scores) != 1 or scores[0] not in (-1, 0, 1):
                raise ValueError('Expected one Classic candidate score')
            score = scores[0]
            counts['games'] += 1
            counts['wins' if score > 0 else 'losses' if score < 0 else 'draws'] += 1
            summary['completed'] += 1
            cost = episode['cost_usd']
            if cost is not None:
                summary['cost_usd'] += cost
            summary['episodes'].append({'episode_id': eid, 'episode_request_id': episode['id'],
                                        'request_id': state['id'], 'opponent_seat': label,
                                        'score': score, 'cost_usd': cost})
        if counts['games'] != state['completed_count']:
            raise ValueError('Current request episode list does not cover its completed count')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', default=SERVER)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('submit')
    create.add_argument('--policy', required=True)
    create.add_argument('--opponent', action='append', required=True, help='NAME=IMMUTABLE_POLICY_VERSION_ID')
    create.add_argument('--games-per-opponent', type=int, required=True)
    create.add_argument('--key', required=True)
    create.add_argument('--checkpoint-sha256', required=True)
    create.add_argument('--source-commit', required=True)
    create.add_argument('--image-digest', required=True)
    create.add_argument('--dry-run', action='store_true')
    for name in ('status', 'collect'):
        commands.add_parser(name).add_argument('--output', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'submit':
        pairs = [value.split('=', 1) for value in args.opponent]
        if any(len(pair) != 2 for pair in pairs) or len({p[0] for p in pairs}) != len(pairs):
            parser.error('Specify distinct opponents as NAME=ID')
        panel = prepare(args.policy, dict(pairs), args.games_per_opponent, args.key,
                        args.checkpoint_sha256, args.source_commit, args.image_digest)
        if args.dry_run:
            if args.output.exists():
                if load_panel(args.output) != panel:
                    raise ValueError('Existing intent differs')
            else:
                args.output.mkdir(parents=True)
                for label, body in panel['requests'].items():
                    save(args.output / f'{label}-payload.json', body)
                save(args.output / 'intent.json', panel)
            print(json.dumps({'total_episodes': panel['total_episodes'], 'requests': len(panel['requests']),
                              'intent': str(args.output / 'intent.json')}))
        else:
            submit(args.output, panel, Observatory(args.server))
    else:
        states = status(args.output, Observatory(args.server))
        if args.command == 'collect':
            result = summarize(load_panel(args.output), states)
            save(args.output / 'summary.json', result)
            print(json.dumps({k: result[k] for k in ('completed', 'failed', 'pending', 'cost_usd')}))
        else:
            print(json.dumps([{k: s[k] for k in ('id', 'status', 'completed_count', 'failed_count')}
                              for s in states.values()]))


if __name__ == '__main__':
    main()

"""Replay saved arena rows, recording complete trajectories and observed failures.

Example: JAX_PLATFORMS=cpu python -m generals.evaluation.replay RUN_DIR --max-samples 4
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals.core import game
from .arena import Rules, action_counters, transition
from .cli import agent

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('passes', 'splits', 'build_attempts', 'builds', 'invalid_moves', 'malformed_commands')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_rows(rows, *, max_samples=1, result='loss', **filters):
    """Select across suite/opponent groups before taking another row per group."""
    groups = defaultdict(list)
    for row in rows:
        if result != 'any' and row['result'] != result:
            continue
        if any(value is not None and str(row[name]) != str(value) for name, value in filters.items()):
            continue
        groups[(row['suite'], row['opponent'])].append(row)
    selected = []
    while groups and len(selected) < max_samples:
        for key in list(groups):
            selected.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(selected) == max_samples:
                break
    return selected


def load_grid(run_dir, row):
    """Recover the stored board and original general-label permutation."""
    with np.load(Path(run_dir) / f"{row['suite']}_boards.npz", allow_pickle=False) as boards:
        grid = boards[str(row['board_id'])].copy()
    if int(row['swapped']):
        grid = np.where(grid == 1, 2, np.where(grid == 2, 1, grid))
    if grid.shape != (int(row['height']), int(row['width'])):
        raise ValueError('Stored board shape does not match CSV')
    return grid


def rules_for(metadata, suite):
    if suite in metadata.get('suite_rules', {}):
        return Rules(**metadata['suite_rules'][suite]), 'stored suite_rules'
    # Legacy files predate explicit rules metadata. No board generation occurs.
    defaults = {'open4': Rules(), 'terrain4': Rules(), 'classic8': Rules(800),
                'classic12': Rules(800), 'competition': Rules(1200, True, 800)}
    if suite not in defaults:
        raise ValueError(f'No stored rules for unknown suite {suite}')
    return defaults[suite], 'legacy suite defaults; historical rules provenance unverified'


def make_policy(name, rules, checkpoint=None, source=None):
    if source:
        if name != 'sentinel':
            raise ValueError('--candidate-source currently supports Sentinel snapshots only')
        spec = importlib.util.spec_from_file_location('generals.agents._sentinel_snapshot', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        player = module.SentinelAgent(build_castles=rules.build_castles,
                                      deathtouch_turn=rules.deathtouch_turn, max_turns=rules.max_turns)
        return player.act, player.decision
    policy = agent(name, rules, checkpoint)
    owner = getattr(policy, '__self__', None)
    return policy, getattr(owner, 'decision', None)


def replay_game(grid, candidate, opponent, rules, *, seat, action_seed, decision=None):
    """Record full state only outside policies; policies receive fog observations.

    Keys split in the exact candidate/opponent order used by arena.make_runner.
    Arrays hold T+1 states, T observations/actions/keys/counters and telemetry.
    """
    @jax.jit
    def step(state, key):
        keys = jax.random.split(key, 3)
        ours = game.get_observation(state, seat)
        theirs = game.get_observation(state, 1 - seat)
        action = candidate(ours, keys[1])
        telemetry = {} if decision is None else decision(ours, keys[1])[1]
        enemy_action = opponent(theirs, keys[2])
        actions = jnp.stack((action, enemy_action) if seat == 0 else (enemy_action, action))
        observations = (ours, theirs) if seat == 0 else (theirs, ours)
        after, info = transition(state, actions, rules)
        counters = action_counters(state, after, actions, rules)
        return after, keys[0], observations, actions, counters, telemetry, info.is_done

    state = game.create_initial_state(jnp.array(grid))
    key = jax.random.PRNGKey(action_seed)
    states = [jax.device_get(state)]
    records = []
    finished = False
    while int(state.time) < rules.max_turns and not finished:
        old_key = np.asarray(key)
        state, key, obs, actions, counters, telemetry, terminal = step(state, key)
        state_host, obs, actions, counters, telemetry = jax.device_get((state, obs, actions, counters, telemetry))
        states.append(state_host)
        records.append((old_key, obs, actions, counters, telemetry))
        finished = bool(terminal)
    arrays = {'initial_grid': np.asarray(grid), 'final_key': np.asarray(key)}
    for field in state._fields:
        arrays[f'state_{field}'] = np.stack([getattr(s, field) for s in states])
    arrays['keys'] = np.stack([r[0] for r in records]) if records else np.empty((0, 2), np.uint32)
    arrays['actions'] = np.stack([r[2] for r in records]) if records else np.empty((0, 2, 5), np.int32)
    arrays['action_counters'] = np.stack([r[3] for r in records]) if records else np.empty((0, 2, len(METRICS)), np.int32)
    if records:
        for player in range(2):
            for field in records[0][1][player]._fields:
                if getattr(records[0][1][player], field) is not None:
                    arrays[f'observation_{player}_{field}'] = np.stack([getattr(r[1][player], field) for r in records])
        for field in records[0][4]:
            arrays[f'telemetry_{field}'] = np.stack([r[4][field] for r in records])
    winner = int(state.winner)
    outcome = 'draw' if winner < 0 else ('win' if winner == seat else 'loss')
    result = dict(result=outcome, winner=winner, turns=int(state.time), terminal=finished)
    totals = arrays['action_counters'].sum(axis=0)
    for prefix, player in (('own', seat), ('enemy', 1 - seat)):
        result.update({f'{prefix}_{metric}': int(value) for metric, value in zip(METRICS, totals[player])})
    return arrays, result


def compare_result(actual, expected):
    mismatch = {}
    for name, value in actual.items():
        if name not in expected:
            continue
        recorded = expected[name]
        if isinstance(value, bool):
            recorded = str(recorded).lower() == 'true'
        elif isinstance(value, int):
            recorded = int(recorded)
        if value != recorded:
            mismatch[name] = {'recorded': recorded, 'replayed': value}
    return mismatch


def diagnostics(arrays, result, seat, rules, last_turns=20):
    turns = len(arrays['actions'])
    history = []
    for t in range(turns):
        own = arrays['state_ownership'][t, seat]
        enemy = arrays['state_ownership'][t, 1 - seat]
        next_own = arrays['state_ownership'][t + 1, seat]
        general = own & arrays['state_generals'][t]
        armies = arrays['state_armies'][t]
        visible_enemy = arrays[f'observation_{seat}_opponent_cells'][t]
        observed_armies = arrays[f'observation_{seat}_armies'][t]
        home = np.argwhere(general)
        threat = 0
        if len(home):
            distance = np.abs(np.indices(own.shape)[0] - home[0, 0]) + np.abs(np.indices(own.shape)[1] - home[0, 1])
            threat = int(np.max(np.where(visible_enemy & (distance == 1), np.maximum(observed_armies - 1, 0), 0)))
        telemetry = {key.removeprefix('telemetry_'): value[t].item()
                     for key, value in arrays.items() if key.startswith('telemetry_') and value[t].ndim == 0}
        before_castles = arrays['state_castles'][t]
        after_castles = arrays['state_castles'][t + 1]
        row = dict(turn=int(arrays['state_time'][t]), own_action=arrays['actions'][t, seat].tolist(),
                   enemy_action=arrays['actions'][t, 1 - seat].tolist(),
                   general_army=int(armies[general].sum()), visible_adjacent_threat=threat,
                   own_army=int(armies[own].sum()), enemy_army=int(armies[enemy].sum()),
                   own_land=int(own.sum()), enemy_land=int(enemy.sum()),
                   own_castles=int((own & before_castles).sum()),
                   castle_gains=int((next_own & after_castles & ~(own & before_castles)).sum()),
                   castle_losses=int((own & before_castles & ~(next_own & after_castles)).sum()),
                   land_gains=int((next_own & ~own).sum()), land_losses=int((own & ~next_own).sum()),
                   telemetry=telemetry)
        history.append(row)
    if not result['terminal']:
        explanation = f"Turn limit {rules.max_turns} reached; arena records a draw regardless of army or land lead."
    elif result['winner'] < 0:
        explanation = 'Terminal draw: no winning team recorded (inspect final eliminated/general state).'
    else:
        loser = 1 - result['winner']
        explanation = f"Player {loser} eliminated; winning player {result['winner']}."
        if rules.deathtouch_turn is not None and history and history[-1]['turn'] >= rules.deathtouch_turn:
            explanation += ' Deathtouch was active on the terminal move; reaching a general can capture it regardless of defending army.'
    categories = []
    if result['result'] != 'win' and history:
        if all(row['own_action'][0] == 1 for row in history):
            categories.append('passed_every_turn')
        if max(row['own_land'] for row in history) == 1:
            categories.append('never_expanded_beyond_general')
    if result['result'] == 'loss':
        categories.append('terminal_defeat')
        if history and history[-1]['visible_adjacent_threat'] > 0:
            categories.append('visible_adjacent_enemy_before_defeat')
        if history and history[-1]['own_action'][0] == 1:
            categories.append('passed_on_terminal_turn')
        if history and history[-1]['own_army'] < history[-1]['enemy_army']:
            categories.append('army_deficit_before_defeat')
        if any(r['castle_losses'] for r in history[-last_turns:]):
            categories.append('castle_loss_in_final_window')
        if any(r['general_army'] < r['telemetry'].get('general_reserve', 0) for r in history[-last_turns:]):
            categories.append('below_policy_general_reserve_in_final_window')
    elif not result['terminal']:
        categories.append('timeout')
        if history and history[-1]['own_army'] > history[-1]['enemy_army']:
            categories.append('timeout_with_army_lead')
    streak = longest = 0
    for row in history:
        streak = streak + 1 if row['own_action'][0] == 1 else 0
        longest = max(longest, streak)
    return dict(termination=explanation, observed_categories=categories, final_turns=history[-last_turns:],
                activity=dict(longest_pass_streak=longest, final_pass_streak=streak,
                              maximum_owned_land=max((row['own_land'] for row in history), default=0),
                              visible_enemy_general_turns=sum(bool(row['telemetry'].get('goal_visible_general', False)) for row in history)),
                event_totals={name: sum(r[name] for r in history) for name in ('castle_gains', 'castle_losses', 'land_gains', 'land_losses')},
                interpretation='Categories describe observations; they do not establish causes. Full state metrics are recorder-only, never policy inputs.')


def source_provenance(metadata, row, candidate_source=None):
    recorded = metadata.get('source_hashes', {})
    paths = set(recorded)
    for directory in ('generals/agents', 'generals/core', 'generals/modifiers',
                      'generals/evaluation', 'generals/training', 'examples/_experimental/ppo'):
        paths.update(str(path.relative_to(ROOT)) for path in (ROOT / directory).rglob('*.py'))
    current = {name: file_hash(ROOT / name) if (ROOT / name).is_file() else None for name in sorted(paths)}
    changed = [name for name in recorded if current[name] != recorded[name]]
    candidate_module = {'sentinel': 'sentinel_agent', 'old-ppo': None, 'learned': None}.get(row['candidate'], f"{row['candidate']}_agent")
    critical = [name for name in changed if name.startswith(('generals/core/', 'generals/modifiers/'))
                or name in ('generals/evaluation/arena.py', 'generals/evaluation/scenarios.py')
                or name == f"generals/agents/{row['opponent']}_agent.py"
                or (candidate_module and name == f'generals/agents/{candidate_module}.py' and not candidate_source)
                or (row['candidate'] == 'learned' and name.startswith('generals/training/'))
                or (row['candidate'] == 'old-ppo' and name.startswith('examples/_experimental/ppo/'))]
    candidate_path = Path(candidate_source) if candidate_source else ROOT / 'generals/agents' / f'{candidate_module}.py'
    actual_candidate_hash = file_hash(candidate_path) if candidate_path.is_file() else None
    expected_candidate_hash = metadata.get('candidate_source_sha256')
    if expected_candidate_hash and actual_candidate_hash != expected_candidate_hash:
        critical.append('candidate_source_sha256')
    return dict(status='recorded' if recorded else 'historical_source_hashes_unavailable',
                recorded_source_hashes=recorded, replay_source_hashes=current,
                changed_sources=changed, critical_changed_sources=critical,
                candidate_source=str(candidate_path), candidate_source_sha256=actual_candidate_hash,
                replay_tool_sha256=file_hash(__file__),
                note=metadata.get('provenance_note', metadata.get('retroactive_provenance')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-samples', type=int, default=1)
    parser.add_argument('--last-turns', type=int, default=20)
    parser.add_argument('--result', choices=('loss', 'draw', 'win', 'any'), default='loss')
    parser.add_argument('--suite')
    parser.add_argument('--opponent')
    for name in ('board-id', 'seat', 'swapped', 'repeat'):
        parser.add_argument(f'--{name}', type=int)
    parser.add_argument('--candidate-source', type=Path, help='Explicit Sentinel source snapshot for historical replay')
    parser.add_argument('--allow-mismatch', action='store_true', help='Save changed-source/outcome diagnostics with explicit mismatch labels')
    args = parser.parse_args()
    if min(args.max_samples, args.last_turns) < 1:
        parser.error('sample and turn counts must be positive')
    metadata = json.loads((args.run_dir / 'metadata.json').read_text())
    with (args.run_dir / 'games.csv').open() as stream:
        rows = select_rows(list(csv.DictReader(stream)), max_samples=args.max_samples, result=args.result,
                           suite=args.suite, opponent=args.opponent, board_id=args.board_id,
                           seat=args.seat, swapped=args.swapped, repeat=args.repeat)
    if not rows:
        parser.error('No recorded games match selection')
    output = args.output or args.run_dir / 'replays'
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for row in rows:
        rules, rule_provenance = rules_for(metadata, row['suite'])
        provenance = source_provenance(metadata, row, args.candidate_source)
        checkpoint = metadata.get('checkpoint')
        if checkpoint and metadata.get('checkpoint_sha256') != file_hash(checkpoint):
            provenance['critical_changed_sources'].append('checkpoint_sha256')
        if provenance['critical_changed_sources'] and not args.allow_mismatch:
            raise SystemExit(f"Source changed: {provenance['critical_changed_sources']}. Supply the historical snapshot or explicitly use --allow-mismatch.")
        ours, decision = make_policy(row['candidate'], rules, checkpoint, args.candidate_source)
        theirs, _ = make_policy(row['opponent'], rules)
        arrays, actual = replay_game(load_grid(args.run_dir, row), ours, theirs, rules,
                                     seat=int(row['seat']), action_seed=int(row['action_seed']), decision=decision)
        mismatch = compare_result(actual, row)
        detail = diagnostics(arrays, actual, int(row['seat']), rules, args.last_turns)
        name = f"{row['suite']}-{row['opponent']}-b{row['board_id']}-r{row['repeat']}-s{row['seat']}-swap{row['swapped']}"
        np.savez_compressed(output / f'{name}.npz', **arrays)
        report = dict(recorded_row=row, replayed_result=actual, outcome_and_counter_mismatches=mismatch,
                      verified_recorded_outcome=not mismatch, source_provenance=provenance,
                      rules=asdict(rules), rule_provenance=rule_provenance,
                      runtime=dict(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()]), **detail)
        (output / f'{name}.json').write_text(json.dumps(report, indent=2) + '\n')
        text = [f'# Replay: {name}', '', detail['termination'], '',
                f"Recorded outcome/turn/counters match: **{not mismatch}**. Source provenance: {provenance['status']}.",
                f"Changed critical sources: {provenance['critical_changed_sources']}. Outcome mismatches: {mismatch}.",
                '', detail['interpretation'], '', f"Observed categories: {', '.join(detail['observed_categories']) or 'none'}.",
                '', 'Actions: `[kind, row, column, direction, split]`; kind 0 move, 1 pass, 2 build.', '',
                '| Turn | Own action | Opponent action | General army | Visible adjacent threat | Policy reserve | Armies own/enemy | Castles own (gain/loss) |',
                '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for turn in detail['final_turns']:
            text.append(f"| {turn['turn']} | {turn['own_action']} | {turn['enemy_action']} | {turn['general_army']} | {turn['visible_adjacent_threat']} | {turn['telemetry'].get('general_reserve', 'n/a')} | {turn['own_army']}/{turn['enemy_army']} | {turn['own_castles']} (+{turn['castle_gains']}/-{turn['castle_losses']}) |")
        (output / f'{name}.md').write_text('\n'.join(text) + '\n')
        summaries.append(dict(name=name, matched=not mismatch, result=actual['result'], turns=actual['turns'],
                              critical_changed_sources=provenance['critical_changed_sources'],
                              observed_categories=detail['observed_categories']))
        print(f"{name}: {actual['result']} at turn {actual['turns']}; matches={not mismatch}; {detail['observed_categories']}", flush=True)
        if mismatch and not args.allow_mismatch:
            raise SystemExit(f'Replay mismatch; trace and diagnostic report saved to {output / name}')
    category_counts = Counter(category for row in summaries for category in row['observed_categories'])
    (output / 'summary.json').write_text(json.dumps(dict(samples=summaries, observed_category_counts=category_counts,
        interpretation='Selected representative games, not an unbiased failure-frequency estimate; categories may overlap.'), indent=2) + '\n')


if __name__ == '__main__':
    main()

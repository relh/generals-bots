"""Paired, reproducible evaluation of the 4x4 PPO policy against local bots.

Run as a module from the repository root. No training or auto-reset occurs.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent
from generals.agents.harvester_agent import HarvesterAgent
from generals.core import game
from generals.core.action import compute_valid_move_mask

from .network import PolicyValueNetwork, obs_to_array
from .train import random_action


def policy_action(network, observation, key):
    mask = compute_valid_move_mask(observation.armies, observation.owned_cells, observation.mountains)
    return network(obs_to_array(observation), mask, key)[0]


def make_runner(opponent_action, max_turns):
    """Run independent games together, retaining the first terminal state."""
    @eqx.filter_jit
    def run(network, grids, keys, seats):
        states = jax.vmap(game.create_initial_state)(grids)

        def unfinished(carry):
            states, _ = carry
            return jnp.any((states.winner < 0) & (states.time < max_turns))

        def tick(carry):
            states, keys = carry
            split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
            candidate_obs = jax.vmap(game.get_observation)(states, seats)
            opponent_obs = jax.vmap(game.get_observation)(states, 1 - seats)
            candidate_actions = jax.vmap(lambda o, k: policy_action(network, o, k))(
                candidate_obs, split[:, 1])
            opponent_actions = jax.vmap(opponent_action)(opponent_obs, split[:, 2])
            actions = jnp.stack([
                jnp.where(seats[:, None] == 0, candidate_actions, opponent_actions),
                jnp.where(seats[:, None] == 0, opponent_actions, candidate_actions),
            ], axis=1)
            stepped, _ = jax.vmap(game.step)(states, actions)
            active = (states.winner < 0) & (states.time < max_turns)
            states = jax.tree.map(
                lambda new, old: jnp.where(active.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
                stepped, states,
            )
            return states, split[:, 0]

        states, _ = jax.lax.while_loop(unfinished, tick, (states, keys))
        return states
    return run


def connected(grid):
    """Require all non-mountain cells to be reachable, including both generals."""
    cells = set(map(tuple, np.argwhere(grid != -2)))
    reached = {next(iter(cells))}
    todo = list(reached)
    while todo:
        r, c = todo.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (r + dr, c + dc)
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                todo.append(neighbor)
    return reached == cells


def make_boards(suite, seed, terrain_boards=64):
    if suite == 'open':
        boards = []
        # Exhaust every unordered pair of distinct starting positions.
        for a, b in itertools.combinations(range(16), 2):
            grid = np.zeros(16, dtype=np.int32)
            grid[a], grid[b] = 1, 2
            boards.append(grid.reshape(4, 4))
        return boards
    rng = np.random.default_rng(seed)
    boards = []
    while len(boards) < terrain_boards:
        grid = np.zeros(16, dtype=np.int32)
        positions = rng.choice(16, 6, replace=False)
        grid[positions[:2]] = [1, 2]
        grid[positions[2:4]] = -2
        grid[positions[4:]] = rng.integers(5, 13, size=2)
        grid = grid.reshape(4, 4)
        if connected(grid):
            boards.append(grid)
    return boards


def make_cases(boards, seed, repeats):
    rng = np.random.default_rng(seed)
    cases = []
    for board_id, grid in enumerate(boards):
        for repeat in range(repeats):
            action_seed = int(rng.integers(0, 2**31))
            for label_swap in (0, 1):
                board = np.where(grid == 1, 2, np.where(grid == 2, 1, grid)) if label_swap else grid
                for seat in (0, 1):
                    cases.append(dict(board_id=board_id, repeat=repeat, label_swap=label_swap,
                                      seat=seat, action_seed=action_seed, grid=board))
    return cases


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(rows):
    wins = sum(row['result'] == 'win' for row in rows)
    losses = sum(row['result'] == 'loss' for row in rows)
    draws = len(rows) - wins - losses
    groups = {}
    for row in rows:
        groups.setdefault(row['board_id'], []).append(row)
    board_wins = [np.mean([r['result'] == 'win' for r in group]) for group in groups.values()]
    board_scores = [np.mean([r['score'] for r in group]) for group in groups.values()]
    rng = np.random.default_rng(731)
    return dict(games=len(rows), wins=wins, losses=losses, draws=draws,
                win_rate=wins / len(rows), score=(wins + 0.5 * draws) / len(rows),
                win_rate_ci95=interval(board_wins, rng), score_ci95=interval(board_scores, rng),
                mean_turns=float(np.mean([r['turns'] for r in rows])),
                seat_win_rates={str(seat): float(np.mean([r['result'] == 'win' for r in rows if r['seat'] == seat]))
                                for seat in (0, 1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--terrain-boards', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-turns', type=int, default=500)
    parser.add_argument('--suites', nargs='+', choices=['open', 'terrain'], default=['open', 'terrain'])
    parser.add_argument('--opponents', nargs='+', default=['training_random', 'random', 'expander', 'hunter', 'harvester'])
    args = parser.parse_args()
    if min(args.repeats, args.terrain_boards, args.batch_size, args.max_turns) < 1:
        parser.error('counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    # Reconstruct the exact initialization used by train.py (seed 42, split once).
    initial = PolicyValueNetwork(jax.random.split(jax.random.PRNGKey(42))[1])
    trained = eqx.tree_deserialise_leaves(args.model, initial)
    opponents = dict(training_random=lambda o, k: random_action(k, o),
                     random=RandomAgent().act, expander=ExpanderAgent().act,
                     hunter=HunterAgent().act, harvester=HarvesterAgent().act)
    metadata = dict(model=str(args.model.resolve()), sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    seed=args.seed, repeats=args.repeats, terrain_boards=args.terrain_boards,
                    max_turns=args.max_turns, devices=[str(d) for d in jax.devices()],
                    jax_version=jax.__version__, numpy_version=np.__version__,
                    rules='base game, fog of war, no modifiers; timeout is a draw',
                    interval='95% percentile bootstrap over boards, 5000 resamples',
                    pairing='both spawn assignments and both player IDs; same action seeds for both models',
                    terrain='4x4, two mountains, two castles with 5-12 armies, connected non-mountain cells')
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    summaries = {}
    fields = ['suite', 'opponent', 'candidate', 'board_id', 'repeat', 'label_swap', 'seat',
              'action_seed', 'result', 'score', 'winner', 'turns', 'own_land', 'enemy_land', 'own_army', 'enemy_army']
    with (args.output / 'games.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for suite in args.suites:
            boards = make_boards(suite, args.seed, args.terrain_boards)
            np.save(args.output / f'{suite}_boards.npy', np.stack(boards))
            cases = make_cases(boards, args.seed + (0 if suite == 'open' else 1), args.repeats)
            for opponent in args.opponents:
                run = make_runner(opponents[opponent], args.max_turns)
                matchup_rows = {}
                for candidate, network in [('trained', trained), ('initial', initial)]:
                    rows = []
                    start = time.monotonic()
                    print(f'Starting {suite} / {opponent} / {candidate}: {len(cases)} games', flush=True)
                    for offset in range(0, len(cases), args.batch_size):
                        batch = cases[offset:offset + args.batch_size]
                        states = run(network, jnp.asarray(np.stack([c['grid'] for c in batch])),
                                     jnp.stack([jax.random.PRNGKey(c['action_seed']) for c in batch]),
                                     jnp.array([c['seat'] for c in batch]))
                        states = jax.device_get(states)
                        for i, case in enumerate(batch):
                            seat = case['seat']
                            winner = int(states.winner[i])
                            result = 'draw' if winner < 0 else ('win' if winner == seat else 'loss')
                            armies = states.armies[i]
                            ownership = states.ownership[i]
                            row = {k: v for k, v in case.items() if k != 'grid'}
                            row.update(suite=suite, opponent=opponent, candidate=candidate,
                                       result=result, score={'win': 1.0, 'draw': 0.5, 'loss': 0.0}[result],
                                       winner=winner, turns=int(states.time[i]),
                                       own_land=int(ownership[seat].sum()), enemy_land=int(ownership[1-seat].sum()),
                                       own_army=int((armies * ownership[seat]).sum()),
                                       enemy_army=int((armies * ownership[1-seat]).sum()))
                            writer.writerow(row)
                            rows.append(row)
                        output.flush()
                    summary = summarize(rows)
                    summary['wall_seconds'] = time.monotonic() - start
                    summaries[f'{suite}/{opponent}/{candidate}'] = summary
                    matchup_rows[candidate] = rows
                    print(f'{suite}/{opponent}/{candidate}: {summary}', flush=True)
                # Paired improvement: aggregate corresponding games within each board.
                board_deltas = {}
                for trained_row, initial_row in zip(matchup_rows['trained'], matchup_rows['initial']):
                    board_deltas.setdefault(trained_row['board_id'], []).append(trained_row['score'] - initial_row['score'])
                deltas = [np.mean(v) for v in board_deltas.values()]
                summaries[f'{suite}/{opponent}/improvement'] = dict(
                    score_delta=float(np.mean(deltas)),
                    score_delta_ci95=interval(deltas, np.random.default_rng(987)))
                (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print('Evaluation complete.', flush=True)


if __name__ == '__main__':
    main()

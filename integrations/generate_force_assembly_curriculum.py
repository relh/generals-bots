"""Teacher-free frontier starts reached by legal play on fresh Classic maps."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.harvester_agent import ExpanderHarvesterAgent
from generals.agents.sentinel_agent import SentinelAgent
from generals.core import coworld_game as engine
from generals.core.grid import generate_grid
from integrations.classic_contract import verify_engine
from integrations.classic_position_curriculum import load_positions

DOMAIN = "force-assembly-v1"
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)


def map_seed(root, split, index):
    return int.from_bytes(hashlib.sha256(f"{DOMAIN}:{split}:{root}:{index}".encode()).digest()[:4], "little")


@jax.jit
def initial(seed, dimensions):
    grid = generate_grid(jax.random.PRNGKey(seed), grid_dims=(21, 21),
                         playable_dims=dimensions, pad_to=21,
                         mountain_density_range=(0.24, 0.26), min_generals_distance=17,
                         num_castles_range=(9, 11), castle_val_range=(40, 51))
    return engine.create_initial_state(grid), grid


def public_metrics(obs):
    """Select visible pressure with useful owned merge routes; no hidden inputs."""
    own = obs.owned_cells
    armies = obs.armies
    maximum = jnp.max(jnp.where(own, armies, 0))
    visible_enemy = jnp.max(jnp.where(obs.opponent_cells, armies, 0))
    own_army = jnp.sum(jnp.where(own, armies, 0))
    reserve_stacks = jnp.sum(own & (armies >= 4))
    joins = jnp.int32(0)
    immediate = jnp.array(False)
    for axis, shift, edge in ((0, 1, 0), (0, -1, -1), (1, 1, 0), (1, -1, -1)):
        destination_owned = jnp.roll(own, shift, axis)
        destination_army = jnp.roll(armies, shift, axis)
        neighboring_enemy = jnp.roll(obs.opponent_cells, shift, axis)
        if axis == 0:
            destination_owned = destination_owned.at[edge, :].set(False)
            neighboring_enemy = neighboring_enemy.at[edge, :].set(False)
        else:
            destination_owned = destination_owned.at[:, edge].set(False)
            neighboring_enemy = neighboring_enemy.at[:, edge].set(False)
        joins += jnp.sum(own & destination_owned & (armies >= 4) & (destination_army >= 4)
                         & (armies + destination_army - 1 > maximum))
        immediate |= jnp.any(own & obs.generals & neighboring_enemy & (destination_army - 1 > armies))
    qualifies = ((visible_enemy >= 12) & (visible_enemy > maximum)
                 & (2 * own_army >= 3 * visible_enemy) & (reserve_stacks >= 2)
                 & (joins > 0) & ~immediate)
    return jnp.stack((qualifies.astype(jnp.int32), maximum, visible_enemy, own_army,
                      reserve_stacks, joins, immediate.astype(jnp.int32)))


public_snapshot = jax.jit(jax.vmap(lambda state: jnp.stack([
    public_metrics(engine.get_observation(state, side)) for side in (0, 1)])))


def make_advance():
    harvester = ExpanderHarvesterAgent()
    sentinel = SentinelAgent(build_castles=False, deathtouch_turn=None, max_turns=2000)

    def one(state, seed, index):
        observations = [engine.get_observation(state, side) for side in (0, 1)]
        key = jax.random.fold_in(jax.random.PRNGKey(seed), state.time)
        commands = jnp.stack([jax.lax.cond(index % 2 == side,
                            lambda _: harvester.act(observations[side], key),
                            lambda _: sentinel.act(observations[side], key), None) for side in (0, 1)])
        commands = jnp.where(state.winner >= 0, PASS[None], commands)
        valid = jnp.stack([(commands[side, 0] == 1) | engine._move_geometry(state, side, commands[side])[0]
                           for side in (0, 1)])
        next_state, _ = engine.step(state, commands, general_trade=False)
        return next_state, commands, valid

    return jax.jit(jax.vmap(one))


def generate_split(output, *, root, split, maps, max_turns, start_turn, stride, excluded_seeds=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seeds = np.array([map_seed(root, split, i) for i in range(maps)], np.uint32)
    if len(set(map(int, seeds))) != maps or set(map(int, seeds)) & set(excluded_seeds):
        raise ValueError("Generated seeds collide with excluded or duplicate maps")
    dimensions = np.array([[18 + i % 4, 18 + (i // 4) % 4] for i in range(maps)], np.int32)
    states, grids = jax.jit(jax.vmap(initial))(seeds, dimensions)
    map_hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in np.asarray(grids)]
    if len(set(map_hashes)) != maps:
        raise ValueError("Generated maps must be distinct")
    advance = make_advance()
    selected, provenance, history = [], [], []
    began = time.monotonic()
    for turn in range(max_turns):
        states, commands, legal = advance(states, seeds, np.arange(maps, dtype=np.int32))
        if not np.asarray(legal).all():
            raise ValueError(f"Script emitted an illegal pre-step action at tick {turn}")
        history.append(np.asarray(commands))
        if turn + 1 < start_turn or (turn + 1 - start_turn) % stride:
            continue
        metrics = np.asarray(public_snapshot(states))
        live = np.asarray(states.winner) == -1
        rows = np.flatnonzero(live & metrics[:, :, 0].any(axis=1))
        for index in rows:
            state = jax.tree.map(lambda x: np.asarray(x[index]), states)
            selected.append(state)
            provenance.append(dict(map_seed=int(seeds[index]), map_sha256=map_hashes[index],
                                   dimensions=dimensions[index].tolist(), tick=turn + 1,
                                   harvester_seat=int(index % 2),
                                   public_metrics=metrics[index].tolist()))
    if not selected:
        raise ValueError("No legally reached states satisfy the force-assembly criteria")
    if len(selected) > 4096:
        raise ValueError("Position archive exceeds the existing loader limit")
    archive = output / "positions.npz"
    np.savez_compressed(archive, **{name: np.stack([getattr(x, name) for x in selected])
                                  for name in engine.GameState._fields})
    position_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    load_positions(archive, position_sha)
    trace = output / "legal-trajectories.npz"
    np.savez_compressed(trace, actions=np.stack(history), seeds=seeds, dimensions=dimensions,
                        grids=np.asarray(grids))
    repository = Path(__file__).resolve().parents[1]
    sources = sorted((repository / "generals").rglob("*.py")) + [Path(__file__).resolve()]
    manifest = dict(schema="classic-midgame-positions-v1", engine_sha256=verify_engine(),
                    positions_sha256=position_sha, count=len(selected), provenance=provenance,
                    root_training_seed=root, split=split, teacher_labels_enabled=False,
                    map_seeds=list(map(int, seeds)), map_hashes=map_hashes,
                    excluded_map_seeds=sorted(excluded_seeds),
                    source_sha256={str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    trajectories_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    generation_wall_seconds=time.monotonic() - began,
                    scope="Fresh legally reached Classic pressure states; no replay inputs or training targets. Separate action traces are legality evidence only, never loaded by training.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit_split(output):
    """Independently reconstruct every retained position from its legal trace."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["engine_sha256"] != verify_engine() or manifest["teacher_labels_enabled"] is not False:
        raise ValueError("Expected pinned teacher-free Classic positions")
    repository = Path(__file__).resolve().parents[1]
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Generation source differs from its recorded closure")
    trace = output / "legal-trajectories.npz"
    if hashlib.sha256(trace.read_bytes()).hexdigest() != manifest['trajectories_sha256']:
        raise ValueError("Legal trace checksum differs")
    positions = load_positions(output / "positions.npz", manifest['positions_sha256'])
    with np.load(trace, allow_pickle=False) as saved:
        commands, seeds, dimensions, grids = [saved[n] for n in ['actions', 'seeds', 'dimensions', 'grids']]
    if list(map(int, seeds)) != manifest['map_seeds'] or any(
            int(seed) != map_seed(manifest['root_training_seed'], manifest['split'], i)
            for i, seed in enumerate(seeds)):
        raise ValueError("Map seed does not bind the independent generation split")
    if commands.shape[1:] != (len(seeds), 2, 5) or commands.dtype != np.int32:
        raise ValueError("Legal action trace shape or dtype differs")
    states, regenerated = jax.jit(jax.vmap(initial))(seeds, dimensions)
    if not np.array_equal(np.asarray(regenerated), grids):
        raise ValueError("Map is not the original official-distribution draw")
    if [hashlib.sha256(row.tobytes()).hexdigest() for row in grids] != manifest['map_hashes']:
        raise ValueError("Generated map digest differs")
    lookup = {int(seed): i for i, seed in enumerate(seeds)}
    by_tick = {}
    for row, item in enumerate(manifest['provenance']):
        by_tick.setdefault(item['tick'], []).append((row, lookup[item['map_seed']]))

    @jax.jit
    def replay(current, actions):
        valid = jax.vmap(lambda state, pair: jnp.stack([
            (pair[side, 0] == 1) | engine._move_geometry(state, side, pair[side])[0]
            for side in (0, 1)]))(current, actions)
        result = jax.vmap(lambda state, pair: engine.step(state, pair, general_trade=False)[0])(current, actions)
        return result, valid

    checked = 0
    for tick, actions in enumerate(commands, 1):
        states, valid = replay(states, actions)
        if not np.asarray(valid).all():
            raise ValueError("Recorded trace contains an illegal pre-step action")
        if tick not in by_tick:
            continue
        metrics = np.asarray(public_snapshot(states))
        for row, index in by_tick[tick]:
            if any(not np.array_equal(np.asarray(getattr(states, name)[index]),
                                      np.asarray(getattr(positions, name)[row])) for name in engine.GameState._fields):
                raise ValueError("Stored position is not exactly reached by its legal trace")
            if not metrics[index, :, 0].any() or metrics[index].tolist() != manifest['provenance'][row]['public_metrics']:
                raise ValueError("Retained position does not satisfy its public selection criterion")
            checked += 1
    if checked != manifest['count'] or checked != len(manifest['provenance']):
        raise ValueError("Not all retained positions were reconstructed")
    return dict(positions_verified=checked, map_count=len(seeds), ticks_replayed=len(commands),
                pre_step_actions_checked=commands.shape[0] * len(seeds) * 2,
                illegal_pre_step_actions=0,
                qualifying_sides={str(side): sum(p['public_metrics'][side][0]
                                                for p in manifest['provenance']) for side in (0, 1)})


def generate(output, *, source_seed, test_seed, maps=16, max_turns=400, start_turn=100, stride=10):
    verify_engine()
    if source_seed == test_seed or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 2**32
                                      for x in (source_seed, test_seed)):
        raise ValueError("Source/test require distinct unsigned root seeds")
    if (any(isinstance(x, bool) or not isinstance(x, int) for x in (maps, max_turns, start_turn, stride))
            or not 2 <= maps <= 128 or not 1 <= start_turn <= max_turns < 2000
            or not 1 <= stride <= max_turns
            or maps * ((max_turns - start_turn) // stride + 1) > 4096):
        raise ValueError("Require bounded map count, live Classic ticks and sampling stride")
    output = Path(output)
    if output.exists():
        raise ValueError("Curriculum output must be a new directory")
    source = generate_split(output / "source", root=source_seed, split="source", maps=maps,
                            max_turns=max_turns, start_turn=start_turn, stride=stride)
    test = generate_split(output / "test", root=test_seed, split="test", maps=maps,
                          max_turns=max_turns, start_turn=start_turn, stride=stride,
                          excluded_seeds=source['map_seeds'])
    if set(source['map_hashes']) & set(test['map_hashes']):
        raise ValueError("Source/test map hashes intersect")
    return dict(source_positions=source['count'], test_positions=test['count'],
                source_manifest_sha256=hashlib.sha256((output / 'source/manifest.json').read_bytes()).hexdigest(),
                test_manifest_sha256=hashlib.sha256((output / 'test/manifest.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-seed', type=int, required=True)
    parser.add_argument('--test-seed', type=int, required=True)
    parser.add_argument('--maps', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(generate(args.output, source_seed=args.source_seed, test_seed=args.test_seed, maps=args.maps)))


if __name__ == '__main__':
    main()

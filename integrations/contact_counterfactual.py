"""One-shot public action proposals, paired terminal outcomes, no learning."""
import argparse
import hashlib
import json
import resource
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integrations.analyze_spatial_population_pair import cluster_interval
from integrations.spatial_policy_bundle import SpatialPlayerPolicy

PLAN_PATH = Path(__file__).with_name('contact_counterfactual_plan.json')
PLAN = json.loads(PLAN_PATH.read_text())
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.asarray(value)
        digest.update(str((name, value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def initial_map_hash(state):
    return array_hash({f: state[f] for f in ('armies','ownership','ownership_neutral','generals',
                                           'castles','mountains','passable','general_positions','teams')})


def propose(values, mask, probabilities, baseline):
    """Use public planes only. Army log encoding round-trips integer armies."""
    board = values.reshape(16, 21, 21)
    army = np.rint(np.expm1(board[0] * 8)).astype(np.int64)
    mine = board[4] > .5
    largest = int(np.max(np.where(mine, army, 0)))
    baseline_source = baseline % 441 if baseline < 3528 else -1
    alternatives, merges = [], []
    for action in np.flatnonzero(mask[:3528]):
        split, move = divmod(int(action), 1764)
        direction, cell = divmod(move, 441)
        row, col = divmod(cell, 21)
        if board[7 + direction, row, col] < .5:
            continue
        if cell != baseline_source:
            alternatives.append(int(action))
        dr, dc = DIRECTIONS[direction]
        tr, tc = row + dr, col + dc
        if not (0 <= tr < 21 and 0 <= tc < 21 and mine[row, col] and mine[tr, tc]):
            continue
        sent = int(army[row, col]) // 2 if split else int(army[row, col]) - 1
        gain = int(army[tr, tc]) + sent - largest
        if sent > 0 and gain > 0:
            merges.append((gain, int(action)))
    alternative = max(alternatives, key=lambda a: (probabilities[a], -a)) if alternatives else baseline
    merge = max(merges, key=lambda item: (item[0], probabilities[item[1]], -item[1]))[1] if merges else baseline
    hold = 3528 if mask[3528] else baseline
    actions = np.asarray((baseline, alternative, merge, hold), np.int32)
    if not mask[actions].all():
        raise ValueError('Public proposal was illegal')
    return actions, dict(available=[True, bool(alternatives), bool(merges), bool(mask[3528])],
                         distinct_actions=int(len(set(actions.tolist()))),
                         probabilities=probabilities[actions].tolist(),
                         maximum_stack_growth=max((gain for gain, _ in merges), default=0))


def summarize(outcomes, hashes, labels, sides, available):
    """Replicas are repeated outcomes within a map, never independent maps."""
    if outcomes.shape != (256, 4, 4) or not np.isin(outcomes, (-1, 0, 1)).all():
        raise ValueError('Need all 256 maps, four actions, four signed terminal replicas')
    if len(set(hashes)) != 256:
        raise ValueError('Snapshot maps must be globally unique')
    contrasts = {}
    for action in (1, 2, 3):
        delta = (outcomes[:, action] - outcomes[:, 0]).mean(axis=1)
        _, ci = cluster_interval(delta, np.asarray(hashes), seed=PLAN['bootstrap_seed'],
                                 resamples=PLAN['bootstrap_resamples'])
        strata = []
        for label in range(2):
            for side in (0, 1):
                chosen = (labels == label) & (sides == side)
                if int(chosen.sum()) != 64:
                    raise ValueError('Each opponent/seat requires64 unique maps')
                strata.append(dict(opponent=label, side=side, maps=64,
                                   mean_signed_delta=float(delta[chosen].mean())))
        contrasts[PLAN['actions'][action]] = dict(mean_signed_delta=float(delta.mean()),
                                                 cluster_ci95=ci, strata=strata,
                                                 available_maps=int(available[:, action].sum()))
    primary = contrasts[PLAN['actions'][2]]
    gate = PLAN['positive_gate']
    positive = (primary['available_maps'] >= gate['minimum_available_merge_maps']
                and primary['mean_signed_delta'] >= gate['minimum_mean_signed_delta']
                and primary['cluster_ci95'][0] > 0
                and all(s['mean_signed_delta'] >= gate['minimum_opponent_seat_mean_delta'] for s in primary['strata']))
    return dict(schema='generals-contact-counterfactual-result-v1', positive_diagnostic=bool(positive),
                primary=PLAN['actions'][2], contrasts=contrasts,
                scope=PLAN['scope'], decision='Diagnostic only; no teacher qualification, training or promotion')


def make_advance(env):
    """Keep terminal state, suppress autoreset and freeze all completed lanes."""
    import jax
    import jax.numpy as jnp
    from integrations.puffer_codec import decode_action
    @jax.jit
    def advance(states, paired, finished):
        def one(state, actions):
            timestep, _ = env.base.env.step(state, jax.vmap(lambda a: decode_action(a, 21))(actions), env.base.pool)
            rewards = jnp.where(timestep.truncated, 0.0, timestep.reward)
            return timestep.last_state, rewards, timestep.terminated | timestep.truncated
        next_states, reward, ended = jax.vmap(one)(states, paired)
        frozen = jax.tree.map(lambda old, new: jnp.where(
            finished.reshape((len(finished),)+(1,)*(new.ndim-1)), old, new), states, next_states)
        return frozen, reward, ended & ~finished
    return advance


def run(args):
    import jax
    import jax.numpy as jnp
    from metta_training.environment import EnvironmentContext
    from integrations.spatial_selfplay import SpatialFrozenOpponentPufferEnvironment
    from integrations.spatial_frozen_sampling import frozen_action_indices

    if jax.devices()[0].platform != 'gpu':
        raise RuntimeError('This bounded evaluation requires a GPU')
    source = SpatialPlayerPolicy(args.source_bundle)
    opponents = [SpatialPlayerPolicy(p) for p in args.opponent_bundles]
    expected = [PLAN['source_policy_sha256'], *PLAN['opponent_policy_sha256']]
    if [p.asset.metadata['policy_sha256'] for p in (source, *opponents)] != expected:
        raise ValueError('Policy identities differ from preregistration')
    seeds = [*PLAN['collection_seeds'], *PLAN['source_sample_seeds'], PLAN['branch_seed']]
    if any(seed in p.asset.metadata['training_seeds'] for p in (source, *opponents) for seed in seeds):
        raise ValueError('Diagnostic seed overlaps training lineage')
    bindings = {str(p): {f: sha(p/f) for f in ('asset.json', 'policy.bin', 'weights.npz', 'spatial-policy.json')}
                for p in (args.source_bundle, *args.opponent_bundles)}
    if list(bindings.values()) != PLAN['bundle_files_sha256']:
        raise ValueError('Actor bundle bytes or samplers differ from preregistered inputs')
    from integrations.classic_contract import verify_engine, ENGINE_SHA256, ENGINE_COMMIT
    verify_engine()
    write(args.output/'input-bindings.json', dict(bundles=bindings, engine_sha256=ENGINE_SHA256, engine_commit=ENGINE_COMMIT))
    write(args.output/'plan.json', PLAN)

    def actor(policy):
        @jax.jit
        def choose(values, masks, keys):
            with jax.default_matmul_precision('highest'):
                outputs = policy._forward(values, jnp)
            return frozen_action_indices(policy, outputs, masks, keys, values)
        return choose

    choose_source = actor(source)
    global_hashes = set()
    all_rows, all_outcomes = [], []
    start = time.monotonic()
    for opponent_id, opponent in enumerate(opponents):
        out = args.output/f'opponent-{opponent_id}'
        out.mkdir()
        seed = PLAN['collection_seeds'][opponent_id]
        env = SpatialFrozenOpponentPufferEnvironment(
            frozen_bundle=args.opponent_bundles[opponent_id],
            context=EnvironmentContext(seed=seed, index=0, mode='train', output=out),
            parallel_games=PLAN['collection_games_per_opponent'], coworld_pool_size=PLAN['map_pool_size'],
            balance_opponent_sides=True, shaping_weight=0.0, reward_scale=1.0,
            terminal_reward_mode='signed', coworld_position_probability=0.0)
        if not env.base.env.coworld_classic_rules:
            raise ValueError('Instantiated engine must use official Classic rules')
        write(out/'engine.json',dict(engine_sha256=ENGINE_SHA256,coworld_classic_rules=env.base.env.coworld_classic_rules,shaping_weight=0.,reward_scale=1.,position_probability=0.))
        choose_opponent = actor(opponent)
        try:
            env.reset_device(f'{seed}:0:0')
            states, keys, sides = env.states, env.keys, np.asarray(env.sides)
            initial = {f: np.asarray(getattr(states, f)) for f in states._fields}
            initial_hashes = [initial_map_hash({f: v[i] for f, v in initial.items()}) for i in range(len(sides))]
            done = np.zeros(len(sides), bool)
            seen_contact = np.zeros_like(done)
            snapshots, initial_maps, rows, counts = [], [], [], [0, 0]
            batch_rows = jnp.arange(len(sides))

            advance = make_advance(env)

            for turn in range(PLAN['collection_turn_cap']+1):
                values, masks = env._observe_both(states)
                own_values, own_masks = values[batch_rows, env.sides], masks[batch_rows, env.sides]
                sample_keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(PLAN['source_sample_seeds'][opponent_id]), turn), len(sides))
                chosen = choose_source(own_values, own_masks, sample_keys)
                # Plane5 is public visible opponent ownership; no hidden state selects proposals.
                contact = np.asarray(own_values[:, 5*441:6*441].any(axis=1)) & ~done & ~seen_contact
                for row in np.flatnonzero(contact):
                    seen_contact[row] = True
                    side = int(sides[row]); map_hash = initial_hashes[row]
                    if counts[side] >= 64 or map_hash in global_hashes:
                        continue
                    x, mask = np.asarray(own_values[row]), np.asarray(own_masks[row], bool)
                    probabilities = np.asarray(source.predict(0, SimpleNamespace(values=x[None], action_masks=mask[None])).probabilities)
                    actions, details = propose(x, mask, probabilities, int(chosen[row]))
                    snapshot = {f: np.asarray(getattr(states, f)[row]) for f in states._fields}
                    state_time = int(snapshot['time'])
                    if state_time != turn or not 0 <= state_time <= 600:
                        raise ValueError('First-episode snapshot time mismatch')
                    snapshots.append(snapshot)
                    initial_maps.append({f:v[row] for f,v in initial.items()})
                    rows.append(dict(opponent=opponent_id, side=side, original_map_sha256=map_hash,
                                     snapshot_sha256=array_hash(snapshot), turn=state_time,
                                     remaining_turns=2000-state_time, public_sha256=array_hash(dict(values=x, mask=mask)),
                                     actions=actions.tolist(), **details, collection_row=int(row),
                                     source_sample_key=np.asarray(sample_keys[row]).tolist(),
                                     opponent_sample_key=np.asarray(keys[row]).tolist()))
                    global_hashes.add(map_hash); counts[side] += 1
                if turn % 100 == 0:
                    print(json.dumps(dict(stage='collect', opponent=opponent_id, turn=turn, selected=counts)), flush=True)
                if counts == [64, 64] or turn == PLAN['collection_turn_cap']:
                    break
                opposing = choose_opponent(values[batch_rows, 1-env.sides], masks[batch_rows, 1-env.sides], keys)
                if not np.asarray(own_masks)[np.arange(len(sides)),np.asarray(chosen)][~done].all() or not np.asarray(masks[batch_rows,1-env.sides])[np.arange(len(sides)),np.asarray(opposing)][~done].all():
                    raise ValueError('Active collection actor selected illegal action')
                paired = jnp.zeros((len(sides), 2), jnp.int32).at[batch_rows, env.sides].set(chosen).at[batch_rows, 1-env.sides].set(opposing)
                states, _, ended = advance(states, paired, jnp.asarray(done))
                done |= np.asarray(ended)
                keys = jax.vmap(lambda key: jax.random.split(key)[0])(keys)
            write(out/'snapshots.json', rows)
            if counts != [64, 64]:
                raise ValueError(f'Insufficient first-contact states within fixed collection: {counts}')
            fields = {f: np.stack([s[f] for s in snapshots]) for f in states._fields}
            np.savez_compressed(out/'snapshots.npz', **fields)
            np.savez_compressed(out/'initial-maps.npz', **{f:np.stack([m[f] for m in initial_maps]) for f in initial})
            # Store exact public inputs separately; full simulation state never goes into an actor.
            snapshot_states = type(states)(**{f: jnp.asarray(v) for f,v in fields.items()})
            public_values, public_masks = env._observe_both(snapshot_states)
            map_sides = np.asarray([r['side'] for r in rows], np.int32)
            np.savez_compressed(out/'public.npz', values=np.asarray(public_values)[np.arange(128), map_sides],
                                masks=np.asarray(public_masks)[np.arange(128), map_sides])
            for i,row in enumerate(rows):
                actual_public=array_hash(dict(values=np.asarray(public_values[i,map_sides[i]]),mask=np.asarray(public_masks[i,map_sides[i]],bool)))
                if actual_public != row['public_sha256']:
                    raise ValueError('Restored snapshot public observation differs')
            # Lane order: map, action, replica. Actions share keys; replicas do not.
            states = jax.tree.map(lambda v: jnp.repeat(v, 16, axis=0), snapshot_states)
            lane_sides = jnp.repeat(jnp.asarray(map_sides), 16)
            lane_rows = jnp.arange(2048)
            actions = jnp.asarray(np.repeat(np.asarray([r['actions'] for r in rows]), 4, axis=1).reshape(-1))
            roots = jnp.tile(jax.random.PRNGKey(PLAN['branch_seed']), (128,1))
            for word in range(0,64,8):
                map_words=jnp.asarray([int(r['original_map_sha256'][word:word+8],16) for r in rows],jnp.uint32)
                roots=jax.vmap(jax.random.fold_in)(roots,map_words)
            replica_roots = jax.vmap(lambda key: jax.vmap(lambda rep: jax.random.fold_in(key, rep))(jnp.arange(4)))(jnp.asarray(roots))
            lane_keys = jnp.tile(replica_roots[:, None, :, :], (1,4,1,1)).reshape(2048,2)
            done = np.zeros(2048, bool); outcomes=np.zeros(2048, np.float32); ended_at=np.full(2048,-1,np.int32)
            for offset in range(int(max(r['remaining_turns'] for r in rows))):
                values, masks = env._observe_both(states)
                # Absolute state time, actor role and replica determine randomness, never action branch.
                time_keys = jax.vmap(jax.random.fold_in)(lane_keys, states.time.astype(jnp.uint32))
                own_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(time_keys)
                opposing_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(time_keys)
                chosen = actions if offset == 0 else choose_source(values[lane_rows,lane_sides], masks[lane_rows,lane_sides], own_keys)
                opposing = choose_opponent(values[lane_rows,1-lane_sides], masks[lane_rows,1-lane_sides], opposing_keys)
                paired = jnp.zeros((2048,2),jnp.int32).at[lane_rows,lane_sides].set(chosen).at[lane_rows,1-lane_sides].set(opposing)
                legal = np.asarray(masks[lane_rows,lane_sides])[np.arange(2048),np.asarray(chosen)]
                opponent_legal = np.asarray(masks[lane_rows,1-lane_sides])[np.arange(2048),np.asarray(opposing)]
                if not legal[~done].all() or not opponent_legal[~done].all():
                    raise ValueError('Active branch selected illegal action')
                states, reward, ended = advance(states,paired,jnp.asarray(done))
                ended=np.asarray(ended); result=np.asarray(reward)[np.arange(2048),np.asarray(lane_sides)]
                outcomes[ended]=result[ended];ended_at[ended]=np.asarray(states.time)[ended];done |= ended
                if offset % 100 == 0:
                    print(json.dumps(dict(stage='branches',opponent=opponent_id,offset=offset,finished=int(done.sum()),elapsed_seconds=time.monotonic()-start)),flush=True)
                if done.all():break
            if not done.all() or not np.isin(outcomes,(-1,0,1)).all() or (ended_at>2000).any():
                raise ValueError('Branch failed to terminate within original episode cap')
            terminal_fields={f:np.asarray(getattr(states,f)) for f in states._fields}
            terminal_hashes=np.asarray([array_hash({f:v[i] for f,v in terminal_fields.items()}) for i in range(2048)])
            np.savez_compressed(out/'branch-results.npz',outcomes=outcomes.reshape(128,4,4),terminal_turns=ended_at.reshape(128,4,4),terminal_hashes=terminal_hashes.reshape(128,4,4),replica_root_keys=np.asarray(replica_roots))
            # Identical forced actions must produce identical entire terminal results under paired keys.
            matrix=outcomes.reshape(128,4,4); hashes=terminal_hashes.reshape(128,4,4)
            for i,row in enumerate(rows):
                for a in range(4):
                    for b in range(a):
                        if row['actions'][a]==row['actions'][b] and not np.array_equal(hashes[i,a],hashes[i,b]):
                            raise ValueError('Identical-action paired branches diverged')
            all_outcomes.append(matrix);all_rows.extend(rows)
        finally:
            env.close()
    result=summarize(np.concatenate(all_outcomes),[r['original_map_sha256'] for r in all_rows],
                     np.asarray([r['opponent'] for r in all_rows]),np.asarray([r['side'] for r in all_rows]),
                     np.asarray([r['available'] for r in all_rows]))
    result.update(plan_sha256=sha(PLAN_PATH),elapsed_seconds=time.monotonic()-start,hardware=str(jax.devices()[0]),jax_version=jax.__version__,training_steps=0)
    write(args.output/'report.json',result)
    write(args.output/'COMPLETED.json',dict(schema='generals-contact-counterfactual-completed-v1',
                                          report_sha256=sha(args.output/'report.json'),positive_diagnostic=result['positive_diagnostic']))
    print(json.dumps(result),flush=True)


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-bundle',type=Path,required=True)
    parser.add_argument('--opponent-bundles',type=Path,nargs=2,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def timeout(signum,frame):
        raise TimeoutError('Preregistered28minute diagnostic execution cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(PLAN['runtime']['execution_minutes']*60)
    try:run(args)
    except BaseException as error:
        write(args.output/'FAILED.json',dict(error=repr(error),plan_sha256=sha(PLAN_PATH)))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':
    main()

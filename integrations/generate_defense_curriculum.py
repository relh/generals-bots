"""Generate independent synthetic Classic defense starts, without replay inputs.

Terrain and general locations come from fresh official-distribution training maps.
Armies and ownership are constructed midgame exercises, not recorded trajectories.
No teacher action is admitted as a training label by this generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_agent import SentinelAgent
from generals.core import coworld_game as engine
from generals.core.grid import generate_grid
from integrations.classic_contract import verify_engine
from integrations.classic_position_curriculum import load_positions
from integrations.classic_siege_native import SOURCE, ClassicSiegeBatch, compile_library
from integrations.puffer_codec import decode_action, encode_coworld_directional_observation
from integrations.spatial_policy_bundle import SpatialPlayerPolicy

OFFSETS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], np.int32)
PASS = jnp.array([1, 0, 0, 0, 0], jnp.int32)


def _seed(seed, index):
    return int.from_bytes(hashlib.sha256(f"defense-v1:{seed}:{index}".encode()).digest()[:4], "little")


@jax.jit
def _map(key, dimensions):
    return generate_grid(
        key,
        grid_dims=(21, 21),
        playable_dims=dimensions,
        pad_to=21,
        mountain_density_range=(0.24, 0.26),
        min_generals_distance=17,
        num_castles_range=(9, 11),
        castle_val_range=(40, 51),
    )


@jax.jit
def _outcomes(state, actions, opponent_action, seat):
    def step(action):
        commands = jnp.stack((action, opponent_action))
        commands = jnp.where(seat == 0, commands, commands[::-1])
        return engine.step(state, commands, general_trade=False)[0].winner

    return jax.vmap(step)(actions)


def legal_actions(state, seat):
    observation = engine.get_observation(state, seat)
    _, mask = encode_coworld_directional_observation(observation)
    indices = np.flatnonzero(np.asarray(mask))
    return observation, indices, jnp.stack([decode_action(int(index), 21) for index in indices])


def construct_position(source, attack_direction, reserve_direction, *, variant, lost=False):
    """Construct a visible attack and reinforcement on existing open terrain."""
    capital = np.asarray(source.general_positions[0])
    attacker, reserve = capital + OFFSETS[attack_direction], capital + OFFSETS[reserve_direction]
    sites = [tuple(attacker), tuple(reserve)]
    if (
        attack_direction == reserve_direction
        or any(np.any(pos < 0) or np.any(pos >= 21) for pos in (attacker, reserve))
        or any(not bool(source.passable[pos]) or bool(source.castles[pos] | source.generals[pos]) for pos in sites)
    ):
        raise ValueError("Defense exercise needs two distinct open neighbors")
    own = np.asarray(source.ownership).copy()
    armies = np.asarray(source.armies).copy()
    own[:, attacker[0], attacker[1]] = [False, True]
    own[:, reserve[0], reserve[1]] = [True, False]
    armies[tuple(capital)] = 3 + variant % 4
    armies[tuple(attacker)] = 15 + variant % 7
    armies[tuple(reserve)] = 2 if lost else 26 + variant % 11
    # A visible expanding stack makes defensive selection compete with expansion.
    extras = np.argwhere(np.asarray(source.passable & ~source.castles & ~source.generals))
    distant = [
        tuple(pos) for pos in extras if min(np.abs(pos - g).sum() for g in np.asarray(source.general_positions)) >= 4
    ]
    distractor = distant[variant % len(distant)]
    own[:, distractor[0], distractor[1]] = [True, False]
    armies[distractor] = 40 + variant % 13
    state = source._replace(
        ownership=jnp.asarray(own),
        armies=jnp.asarray(armies),
        ownership_neutral=source.passable & ~jnp.asarray(own).any(axis=0),
        time=jnp.int32(101 + variant % 100),
    )
    opposite = (1, 0, 3, 2)
    attack = jnp.array([0, *attacker, opposite[attack_direction], 0], jnp.int32)
    reinforcement = jnp.array([0, *reserve, opposite[reserve_direction], 0], jnp.int32)
    return state, attack, reinforcement


def audit_position(state, attack, seat):
    """Synthetic exercise solvability, using exhaustive legal public actions."""
    obs, indices, actions = legal_actions(state, seat)
    _, _, enemy_actions = legal_actions(state, 1 - seat)
    if not np.any(np.all(np.asarray(enemy_actions) == np.asarray(attack), axis=1)):
        raise ValueError("Synthetic attack is not legal under the public mask")
    if not np.any(np.asarray(obs.opponent_cells) & (np.asarray(obs.armies) >= 15)):
        raise ValueError("Threat must be visible to the defending player")
    outcomes = np.asarray(_outcomes(state, actions, attack, jnp.int32(seat)))
    survivors = outcomes != (1 - seat)
    pass_outcome = int(_outcomes(state, PASS[None], attack, jnp.int32(seat))[0])
    return (
        obs,
        indices,
        actions,
        survivors,
        {
            "legal_actions": len(indices),
            "surviving_actions": int(survivors.sum()),
            "pass_loses_capital": pass_outcome == 1 - seat,
            "defendable": bool(survivors.any()),
            "threat_is_visible": True,
        },
    )


def _native_action(opponent, obs):
    shape = (21, 21)
    kinds = np.ones(shape, np.int32)
    for name, value in (("fog_cells", 0), ("structures_in_fog", 5), ("mountains", 2), ("castles", 3), ("generals", 4)):
        kinds[np.asarray(getattr(obs, name))] = value
    owners = np.where(obs.opponent_cells, 2, np.asarray(obs.owned_cells, np.int32))
    grids = np.stack((kinds, owners, np.asarray(obs.armies)), axis=0)[None].astype(np.int32)
    actions, _ = opponent(
        np.array([[21, 21]], np.int32), np.array([obs.timestep], np.int32), grids, opponent.initial_memory(1)
    )
    return actions[0]


def generate(output, *, seed, pairs=32, excluded_seeds=(), bundle=None):
    """Write compatible teacher-free positions and public-view competence probes."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Training seed must be an unsigned 32-bit integer")
    output = Path(output)
    if output.exists():
        raise ValueError("Curriculum output must be a new directory")
    if isinstance(pairs, bool) or not isinstance(pairs, int) or not 1 <= pairs <= 2048:
        raise ValueError("Require between 1 and 2048 paired exercises")
    verify_engine()
    excluded = set(excluded_seeds)
    states, provenance, probes = [], [], []
    seen_seeds, seen_maps = set(), set()
    # Compile only a public-observation opponent; no hidden state crosses its API.
    output.mkdir(parents=True)
    native = ClassicSiegeBatch(compile_library(output / "classic-siege-probe.so"))
    sentinel = SentinelAgent(build_castles=False, deathtouch_turn=None, max_turns=2000)
    policy = SpatialPlayerPolicy(Path(bundle)) if bundle else None
    for index in range(pairs):
        for attempt in range(100):
            map_seed = _seed(seed, index * 100 + attempt)
            if map_seed in excluded or map_seed in seen_seeds:
                continue
            dimensions = np.array([18 + map_seed % 4, 18 + (map_seed // 4) % 4], np.int32)
            grid = _map(jax.random.PRNGKey(map_seed), jnp.asarray(dimensions))
            digest = hashlib.sha256(np.asarray(grid).tobytes()).hexdigest()
            if digest in seen_maps:
                continue
            source = engine.create_initial_state(grid)
            capital = np.asarray(source.general_positions[0])
            directions = []
            for direction, offset in enumerate(OFFSETS):
                neighbor = capital + offset
                if np.all((neighbor >= 0) & (neighbor < 21)):
                    site = tuple(neighbor)
                    if bool(source.passable[site]) and not bool(source.castles[site] | source.generals[site]):
                        directions.append(direction)
            valid = [(attack, reserve) for attack in directions for reserve in directions if attack != reserve]
            if valid:
                break
        else:
            raise ValueError("Unable to find a distinct training map with credible defense geometry")
        seen_seeds.add(map_seed)
        seen_maps.add(digest)
        attack_direction, reserve_direction = valid[index % len(valid)]
        state, attack, _ = construct_position(source, attack_direction, reserve_direction, variant=index)
        lost, _, _ = construct_position(source, attack_direction, reserve_direction, variant=index, lost=True)
        for seat in (0, 1):
            paired = (
                state
                if seat == 0
                else state._replace(ownership=state.ownership[::-1], general_positions=state.general_positions[::-1])
            )
            lost_pair = (
                lost
                if seat == 0
                else lost._replace(ownership=lost.ownership[::-1], general_positions=lost.general_positions[::-1])
            )
            obs, indices, actions, survivors, audit = audit_position(paired, attack, seat)
            _, _, _, _, lost_audit = audit_position(lost_pair, attack, seat)
            if not audit["pass_loses_capital"] or not audit["defendable"] or lost_audit["defendable"]:
                raise ValueError("Defense/lost control exercise classification failed")
            row = {
                "scenario": index,
                "seat": seat,
                "map_seed": map_seed,
                "map_sha256": digest,
                "dimensions": dimensions.tolist(),
                "attack_direction": attack_direction,
                "reserve_direction": reserve_direction,
                "variant": index,
                "audit": audit,
                "lost_control": lost_audit,
            }
            candidate_actions = {
                "sentinel": np.asarray(sentinel.act(obs, jax.random.PRNGKey(map_seed))),
                "classic_siege_padded": _native_action(native, obs),
            }
            probe = {"scenario": index, "seat": seat, "policies": {}}
            for name, action in candidate_actions.items():
                match = np.all(np.asarray(actions) == action, axis=1)
                probe["policies"][name] = {"legal": bool(match.any()), "survives_attack": bool(survivors[match].any())}
            if policy:
                values, mask = encode_coworld_directional_observation(obs)
                probabilities = np.asarray(
                    policy.predict(
                        0, SimpleNamespace(values=[np.asarray(values)], action_masks=[np.asarray(mask)])
                    ).probabilities
                )
                probe["policies"]["cold_policy"] = {
                    "survival_probability": float(probabilities[indices[survivors]].sum())
                }
            states.append(paired)
            provenance.append(row)
            probes.append(probe)
    arrays = {
        name: np.stack([np.asarray(getattr(state, name)) for state in states]) for name in engine.GameState._fields
    }
    archive = output / "positions.npz"
    np.savez_compressed(archive, **arrays)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    load_positions(archive, digest)
    manifest = {
        "schema": "classic-midgame-positions-v1",
        "engine_sha256": verify_engine(),
        "positions_sha256": digest,
        "count": len(states),
        "provenance": provenance,
        "scope": (
            "Independent train-only synthetic defense exercises on fresh Classic maps; "
            "no replay or held-out inputs. Ownership/army placements are exercises, not recorded game trajectories."
        ),
        "root_training_seed": seed,
        "generation_jax_version": jax.__version__,
        "excluded_map_seeds": sorted(excluded),
        "teacher_labels_enabled": False,
        "training_roles": (
            "Paired positions swap the designated defender between seats. The existing reset loader "
            "samples learner seats independently, so training includes both attacker and defender roles."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {
        "positions": len(states),
        "independent_training_maps": len(seen_maps),
        "attack_directions": sorted({row["attack_direction"] for row in provenance}),
        "scope": "One-step survival against the specified visible attack; not full-game strength.",
        "probes": probes,
        "teacher_admission": {},
        "cold_policy_sha256": hashlib.sha256((Path(bundle) / "policy.bin").read_bytes()).hexdigest()
        if bundle
        else None,
        "native_teacher_source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "teacher_labels_enabled": False,
    }
    for name in ("sentinel", "classic_siege_padded"):
        rows = [row["policies"][name] for row in probes]
        rate = sum(row["survives_attack"] for row in rows) / len(rows)
        report["teacher_admission"][name] = {
            "legal_all": all(row["legal"] for row in rows),
            "survival_rate": rate,
            "competent_on_this_probe": rate >= 0.9 and len(rows) >= 64,
        }
    (output / "competence.json").write_text(json.dumps(report, indent=2) + "\n")
    return manifest, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--exclude-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    manifest, report = generate(
        args.output, seed=args.seed, pairs=args.pairs, excluded_seeds=args.exclude_seeds, bundle=args.bundle
    )
    print(json.dumps({"positions": manifest["count"], "teacher_admission": report["teacher_admission"]}, indent=2))


if __name__ == "__main__":
    main()

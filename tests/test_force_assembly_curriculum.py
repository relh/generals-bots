"""Real Classic replay and split boundaries for teacher-free force assembly."""

import json

import jax.numpy as jnp
import pytest
from types import SimpleNamespace

from integrations.generate_force_assembly_curriculum import audit_split, generate, public_metrics


def test_legally_reached_source_test_split(tmp_path):
    output = tmp_path / "curriculum"
    generated = generate(output, source_seed=9401009, test_seed=9401023,
                         maps=4, max_turns=400, start_turn=100, stride=20)
    source = json.loads((output / 'source/manifest.json').read_text())
    test = json.loads((output / 'test/manifest.json').read_text())
    assert not set(source['map_seeds']) & set(test['map_seeds'])
    assert not set(source['map_hashes']) & set(test['map_hashes'])
    for split in ('source', 'test'):
        proof = audit_split(output / split)
        assert proof['positions_verified'] == generated[split + '_positions']
        assert proof['pre_step_actions_checked'] == 3200
        assert proof['illegal_pre_step_actions'] == 0
        assert all(proof['qualifying_sides'].values())
    archive = output / 'source/positions.npz'
    archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        audit_split(output / 'source')


def test_immediate_capital_danger_is_not_a_force_assembly_exercise():
    own = jnp.zeros((21, 21), bool).at[5, 5:10].set(True)
    armies = jnp.zeros((21, 21), jnp.int32).at[5, 5:10].set(5).at[8, 8].set(12)
    enemy = jnp.zeros((21, 21), bool).at[8, 8].set(True)
    generals = jnp.zeros((21, 21), bool).at[5, 5].set(True)
    obs = SimpleNamespace(owned_cells=own, armies=armies, opponent_cells=enemy, generals=generals)
    assert int(public_metrics(obs)[0]) == 1
    obs.armies = obs.armies.at[4, 5].set(12)
    obs.opponent_cells = obs.opponent_cells.at[4, 5].set(True)
    assert int(public_metrics(obs)[0]) == 0


def test_split_roots_must_differ(tmp_path):
    with pytest.raises(ValueError, match='distinct'):
        generate(tmp_path / 'curriculum', source_seed=1, test_seed=1)

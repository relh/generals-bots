"""The prepared causal comparison uses the actual selected source inputs."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from integrations.hard_opponent_weighting import (derive_pair, digest, load_plan,
                                                  validate_probe_receipt, verify_gate)

SOURCE = Path('/tmp/generals-current-policy-input-v2')


def source_configs():
    if not SOURCE.exists():
        pytest.skip('Selected source input is not staged on this machine')
    return (json.loads((SOURCE / 'build-config.json').read_text()),
            json.loads((SOURCE / 'config.json').read_text()))


def test_original_frozen_pool_and_one_factor_training_pair():
    build, run = source_configs()
    plan = load_plan()
    options = build['python_environment']['options']
    frozen = [digest(SOURCE / Path(path).relative_to('/work/input') / 'policy.bin')
              for path in options['frozen_bundles']]
    assert ['frozen_' + value[:12] for value in frozen] == plan['opponent_order'][:10]
    assert frozen[0] != plan['source_checkpoint_sha256']  # no source mirror in either arm
    assert digest(SOURCE / 'assets/cold/policy.bin') == plan['source_checkpoint_sha256']
    control, treatment, matched_run = derive_pair(build, run, plan)
    before = control['python_environment']['options']['opponent_weights']
    after = treatment['python_environment']['options']['opponent_weights']
    assert [(i, old, new) for i, (old, new) in enumerate(zip(before, after, strict=True))
            if old != new] == [(9, 17, 34), (12, 16, 32)]
    assert sum(before) == 71 and sum(after) == 104
    assert (before[9] + before[12]) / sum(before) == 33 / 71
    assert (after[9] + after[12]) / sum(after) == 66 / 104
    assert matched_run['seed'] == 9107441
    assert matched_run['total_timesteps'] == 16_777_216
    assert matched_run['overrides']['vec.total_agents'] == 4096
    assert matched_run['overrides']['train.horizon'] == 128
    assert matched_run['overrides']['train.minibatch_size'] == 8192
    assert matched_run['overrides']['train.replay_ratio'] == 0.5
    assert matched_run['initialize'] == run['initialize']
    assert control['python_environment']['options']['coworld_position_probability'] == 0.25
    assert treatment['python_environment']['options']['coworld_position_probability'] == 0.25


def test_source_mirror_contamination_is_rejected():
    build, run = source_configs()
    build = deepcopy(build)
    build['python_environment']['options']['opponent_weights'][0] = 12
    with pytest.raises(ValueError, match='Original Classic opponent'):
        derive_pair(build, run, load_plan())


def test_qualified_migration_preserves_policy_and_learner_bytes():
    plan = load_plan()
    migrated = Path(plan['migrated_source']['path'])
    if not migrated.exists() or not SOURCE.exists():
        pytest.skip('Migrated source asset is not staged on this machine')
    old = json.loads((SOURCE / 'assets/cold/asset.json').read_text())
    new = json.loads((migrated / 'asset/asset.json').read_text())
    assert digest(migrated / 'proof.json') == plan['migrated_source']['proof_sha256']
    assert digest(migrated / 'asset/asset.json') == plan['migrated_source']['asset_manifest_sha256']
    assert old['model_sha256'] != new['model_sha256'] == plan['migrated_source']['model_sha256']
    assert old['abi_sha256'] == new['abi_sha256']
    assert digest(migrated / 'asset/policy.bin') == digest(SOURCE / 'assets/cold/policy.bin')
    assert digest(migrated / 'asset/policy.bin.learner') == digest(SOURCE / 'assets/cold/policy.bin.learner')
    assert new['learner_sha256'] == plan['source_learner_sha256']


def test_fresh_development_and_confirmation_seeds():
    plan = load_plan()
    dev = plan['development']
    confirm = plan['reserved_independent_confirmation']
    assert len({plan['training']['seed'], dev['map_seed'], dev['sample_seed'],
                dev['bootstrap_seed'], confirm['map_seed'], confirm['sample_seed']}) == 6
    assert (dev['map_seed'], dev['sample_seed']) == (10441701, 10441703)
    assert (confirm['map_seed'], confirm['sample_seed']) == (10442701, 10442703)
    assert plan['required_treatment_probe']['seed'] not in {
        plan['training']['seed'], dev['map_seed'], dev['sample_seed'], confirm['map_seed'], confirm['sample_seed']}


def test_long_run_requires_exact_treatment_probe_receipt(tmp_path):
    plan = load_plan()
    inputs = tmp_path / 'input'
    inputs.mkdir()
    (inputs / 'seal.json').write_text('{}\n')
    receipt = {
        'experiment': plan['experiment'],
        'source_policy_sha256': plan['source_checkpoint_sha256'],
        'input_seal_sha256': digest(inputs / 'seal.json'),
        'opponent_weights': plan['treatment_weights'],
        'probe_seed': plan['required_treatment_probe']['seed'],
        'gpu_model': 'NVIDIA H100 80GB HBM3',
        'audit': {'environment_steps': 4194304, 'steady_sps': 36182,
                  'environment_count': 4096, 'horizon': 128,
                  'minibatch': 8192, 'replay_ratio': 0.5,
                  'epoch_uptime': [1, 2, 3, 4], 'illegal_actions': 0,
                  'reward_audit': {'nonfinite_rewards': 0, 'native_clipped_rewards': 0,
                                   'native_clipped_terminal_rewards': 0},
                  'opponent_counts_by_seat': {name: {'0': 1, '1': 1}
                                              for name in plan['opponent_order']}},
    }
    path = tmp_path / 'QUALIFIED.json'
    path.write_text(json.dumps(receipt))
    assert validate_probe_receipt(path, inputs, plan) == receipt
    for change in ({'opponent_weights': plan['control_weights']},
                   {'gpu_model': 'NVIDIA B300'},
                   {'audit': {**receipt['audit'], 'steady_sps': 29999}},
                   {'audit': {**receipt['audit'], 'reward_audit': {'nonfinite_rewards': 1}}},
                   {'audit': {**receipt['audit'], 'opponent_counts_by_seat': {}}}):
        bad = {**receipt, **change}
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='not qualified'):
            validate_probe_receipt(path, inputs, plan)


def test_both_source_gates_require_complete_evidence():
    plan = load_plan()
    sampler = {'mode': 'structured_sample', 'move_temperature': 0.05,
               'split_temperature': 0.15, 'early_route_temperature': 0.1,
               'early_route_turns': 100, 'route_half_weight': 0.0,
               'full_action_temperature': 1.0, 'neutral_route_bias': 6.0,
               'weak_owned_route_penalty': 4.0, 'doomed_attack_route_penalty': 4.0}
    observed = dict(sampler, log_gap_scale=0.0)
    report = {'gate_mode': 'same_sampler_source',
              'source_sha256': plan['source_checkpoint_sha256'],
              'opponent_sha256': plan['source_checkpoint_sha256'],
              'sampler': observed, 'opponent_sampler': observed,
              'games': 512, 'seat_counts': {'0': 256, '1': 256},
              'wld': [252, 254, 6], 'unique_initial_maps': 320,
              'match_seed': 51231, 'sample_seed': 17441,
              'held_out': True, 'coworld_classic_rules': True}
    verify_gate(report, sampler, plan['source_checkpoint_sha256'], 51231, 17441)
    with pytest.raises(ValueError, match='complete matched sampler'):
        verify_gate(report, sampler, plan['source_checkpoint_sha256'], 10441691, 10441693)
    with pytest.raises(ValueError, match='complete matched sampler'):
        verify_gate({**report, 'seat_counts': {'0': 512, '1': 0}}, sampler,
                    plan['source_checkpoint_sha256'], 51231, 17441)

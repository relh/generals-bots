"""The prepared causal comparison uses the actual selected source inputs."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from integrations.hard_opponent_weighting import derive_pair, digest, load_plan

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


def test_fresh_development_and_confirmation_seeds():
    plan = load_plan()
    dev = plan['development']
    confirm = plan['reserved_independent_confirmation']
    assert len({plan['training']['seed'], dev['map_seed'], dev['sample_seed'],
                dev['bootstrap_seed'], confirm['map_seed'], confirm['sample_seed']}) == 6
    assert (dev['map_seed'], dev['sample_seed']) == (10441701, 10441703)
    assert (confirm['map_seed'], confirm['sample_seed']) == (10442701, 10442703)

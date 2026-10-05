"""Portable serving rejects undeclared samplers and incompatible contracts."""

import hashlib
import json

import numpy as np
import pytest

from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def bundle(path, *, channels=16, sampler=None, factorized=False):
    path.mkdir()
    build = {'config': {
        'fabric': {'options': {'channels': channels, 'context_radius': 1.01},
                   'observation_size': channels * 441, 'action_sizes': [1765, 2] if factorized else [3529]},
        'python_environment': {'options': {'public_scalar_features': True, 'factorized_actions': factorized}},
    }}
    (path / 'build.json').write_text(json.dumps(build))
    (path / 'training.json').write_text(json.dumps({'build': build}))
    (path / 'policy.bin').write_bytes(np.zeros(1, '<f4').tobytes())
    weights = dict(input_kernel=np.zeros((channels, 1), np.float32),
                   context_kernel=np.zeros((3, 3, 1, 1), np.float32),
                   global_kernel=np.zeros((441, 1), np.float32), readout_kernel=np.zeros((1, 3530), np.float32),
                   action_kernel=np.zeros((1, 8), np.float32))
    for prefix, count in [('local', 1), ('context', 1), ('global', 1), ('output', 3530)]:
        weights[prefix + '_weight'] = np.ones(count, np.float32)
        weights[prefix + '_bias'] = np.zeros(count, np.float32)
    np.savez(path / 'weights.npz', **weights)
    manifest = {'schema': 'puffer5-generals-spatial-v1', 'channels': channels,
                'features': 1, 'global_features': 1, 'prior_count': 0,
                'files': {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                          for name in ['build.json', 'training.json', 'policy.bin', 'weights.npz']}}
    if sampler is not None:
        manifest['serving_action_selection'] = sampler
    (path / 'spatial-policy.json').write_text(json.dumps(manifest))
    return path


def test_current_explicit_sampler_loads_and_reset_preserves_reproducible_stream(tmp_path):
    policy = SpatialPlayerPolicy(bundle(tmp_path / 'current', sampler={
        'mode': 'structured_sample', 'move_temperature': 0.05, 'split_temperature': 0.15,
    }))
    assert policy.observation_size == 7056 and policy.action_mode == 'structured_sample'
    policy.reset('same-game')
    first = policy.action_rng.random(16)
    policy.reset('same-game')
    np.testing.assert_array_equal(first, policy.action_rng.random(16))


@pytest.mark.parametrize('channels', [11, 12])
def test_noncanonical_channels_fail_closed(tmp_path, channels):
    with pytest.raises(ValueError, match='16-plane flat'):
        SpatialPlayerPolicy(bundle(tmp_path / 'invalid', channels=channels))


def test_missing_or_argmax_sampler_and_factorized_actions_are_rejected(tmp_path):
    with pytest.raises(KeyError, match='serving_action_selection'):
        SpatialPlayerPolicy(bundle(tmp_path / 'missing'))
    with pytest.raises(ValueError, match='explicit structured_sample'):
        SpatialPlayerPolicy(bundle(tmp_path / 'argmax', sampler={'mode': 'argmax'}))
    with pytest.raises(ValueError, match='16-plane flat'):
        SpatialPlayerPolicy(bundle(tmp_path / 'factorized', factorized=True))

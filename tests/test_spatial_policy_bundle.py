"""Portable serving binds the current native asset and its explicit sampler."""

import hashlib
import json

import numpy as np
import pytest

from integrations.native_spatial_asset import FACTORY, write_asset
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def seal(path):
    manifest = {'schema': 'generals-spatial-policy-v1',
                'files': {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                          for name in ('asset.json', 'policy.bin', 'weights.npz')}}
    (path / 'spatial-policy.json').write_text(json.dumps(manifest))


def bundle(path):
    original = path.with_suffix('.bin')
    original.write_bytes(bytes(570668 * 4))
    fabric = {'factory': FACTORY, 'compiler': 'standard', 'observation_size': 7056,
              'action_sizes': [3529], 'options': {
                  'channels': 16, 'height': 21, 'width': 21, 'features_per_site': 32,
                  'global_features': 32, 'context_radius': 1.01, 'route_prior_strength': .5,
                  'source_army_prior_strength': .25, 'half_prior_scale': .99,
                  'full_split_prior_strength': .125}}
    write_asset(path, fabric=fabric, factory_source_sha256='a' * 64,
                model_sha256='b' * 64, abi_sha256='c' * 64, policy=original,
                sampler={'mode': 'structured_sample', 'move_temperature': .05, 'split_temperature': .15},
                training_seeds=[8842], provenance={
                    'operation': 'source_cleanup', 'reinforcement_learning_steps_added': 0,
                    'abi_proof_sha256': 'd' * 64, 'ancestors': {'original': 'e' * 64}})
    weights = dict(input_kernel=np.zeros((16, 32), np.float32),
                   context_kernel=np.zeros((3, 3, 32, 32), np.float32),
                   global_kernel=np.zeros((441 * 32, 32), np.float32),
                   readout_kernel=np.zeros((32, 3530), np.float32),
                   action_kernel=np.zeros((32, 8), np.float32))
    for prefix, count in [('local', 32), ('context', 32), ('global', 32), ('output', 3530)]:
        weights[prefix + '_weight'] = np.ones(count, np.float32)
        weights[prefix + '_bias'] = np.zeros(count, np.float32)
    for i in range(5):
        weights[f'prior_source_{i}'] = np.zeros(3530, np.int32)
        weights[f'prior_weight_{i}'] = np.zeros(3530, np.float32)
    np.savez(path / 'weights.npz', **weights)
    seal(path)
    return path


def test_current_asset_loads_and_reset_preserves_reproducible_stream(tmp_path):
    policy = SpatialPlayerPolicy(bundle(tmp_path / 'current'))
    assert policy.observation_size == 7056 and policy.action_mode == 'structured_sample'
    assert policy.features == 32 and policy.prior_count == 5
    assert policy.asset.metadata['training_seeds'] == [8842]
    policy.reset('same-game')
    first = policy.action_rng.random(16)
    policy.reset('same-game')
    np.testing.assert_array_equal(first, policy.action_rng.random(16))


@pytest.mark.parametrize('field,value', [('channels', 11), ('channels', 12)])
def test_noncanonical_architecture_fails_closed(tmp_path, field, value):
    path = bundle(tmp_path / 'invalid')
    metadata = json.loads((path / 'asset.json').read_text())
    metadata['fabric']['options'][field] = value
    (path / 'asset.json').write_text(json.dumps(metadata))
    seal(path)
    with pytest.raises(ValueError):
        SpatialPlayerPolicy(path)


@pytest.mark.parametrize('sampler', [None, {'mode': 'argmax'}])
def test_sampler_is_mandatory_current_asset_metadata(tmp_path, sampler):
    path = bundle(tmp_path / 'invalid')
    metadata = json.loads((path / 'asset.json').read_text())
    metadata['sampler'] = sampler
    (path / 'asset.json').write_text(json.dumps(metadata))
    seal(path)
    with pytest.raises(ValueError):
        SpatialPlayerPolicy(path)


def test_old_manifest_and_payload_tampering_are_rejected(tmp_path):
    path = bundle(tmp_path / 'invalid')
    (path / 'policy.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        SpatialPlayerPolicy(path)
    seal(path)
    manifest = json.loads((path / 'spatial-policy.json').read_text())
    manifest['files']['build.json'] = 'a' * 64
    (path / 'spatial-policy.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unexpected spatial policy bundle files'):
        SpatialPlayerPolicy(path)

"""A matched trial must train and evaluate the same authenticated cold policy."""
import hashlib

import pytest

from integrations.policy_trial import Trial
from tests.test_native_spatial_asset import author


def trial(tmp_path):
    manifest, policy, _ = author(tmp_path / 'source')
    item = Trial.__new__(Trial)
    item.source_asset = manifest
    item.bundle = tmp_path / 'bundle'
    item.bundle.mkdir()
    (item.bundle / 'policy.bin').write_bytes(policy.read_bytes())
    run = {'initialize': {'asset': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         'restore_learner': False}}
    return item, run


def test_exact_cold_initializer_required(tmp_path):
    item, run = trial(tmp_path)
    assert item.validate_source_initializer(run)['policy_sha256']
    other = item.source_asset.with_name('other.json')
    other.write_bytes(item.source_asset.read_bytes())
    run['initialize']['asset'] = str(other)
    with pytest.raises(ValueError, match='exact cold asset'):
        item.validate_source_initializer(run)


def test_manifest_digest_and_serving_policy_must_match(tmp_path):
    item, run = trial(tmp_path)
    original = run['initialize']['manifest_sha256']
    run['initialize']['manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exact cold asset'):
        item.validate_source_initializer(run)
    run['initialize']['manifest_sha256'] = original
    (item.bundle / 'policy.bin').write_bytes(bytes(4))
    with pytest.raises(ValueError, match='serving bundle'):
        item.validate_source_initializer(run)


def test_trial_sampler_can_be_published_as_current_asset(tmp_path):
    from integrations.native_spatial_asset import validate_sampler

    manifest, _, _ = author(tmp_path / 'original')
    inputs = tmp_path / 'input'
    (inputs / 'assets/cold').mkdir(parents=True)
    (inputs / 'assets/cold/asset.json').write_bytes(manifest.read_bytes())
    item = Trial(inputs, tmp_path / 'output')
    validate_sampler(item.sampler)
    assert 'log_gap_scale' not in item.sampler

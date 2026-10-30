"""Native assets bind actual bytes and provenance without inventing RL records."""

import json
from pathlib import Path

import pytest

from integrations.learner_checkpoint import LEARNER_HEADER
from integrations.native_spatial_asset import FACTORY, load_asset, sha256, training_contract, write_asset

COUNT = 570668


def fabric():
    return {'factory': FACTORY, 'compiler': 'standard', 'observation_size': 7056, 'action_sizes': [3529],
            'options': {'channels': 16, 'height': 21, 'width': 21, 'features_per_site': 32,
                        'global_features': 32, 'context_radius': 1.01, 'route_prior_strength': .5,
                        'source_army_prior_strength': .25, 'half_prior_scale': .99,
                        'full_split_prior_strength': .125}}


def author(directory: Path, *, learner=False, provenance=None):
    directory.mkdir()
    policy = directory / 'original-policy.bin'
    policy.write_bytes(bytes(COUNT * 4))
    state = directory / 'original.learner' if learner else None
    if state:
        state.write_bytes(LEARNER_HEADER.pack(b'METTAL01', 12, 25165824, COUNT, .0002) + bytes(COUNT * 4))
    overrides = {'train.gamma': .999, 'train.learning_rate': .0002,
                 'train.horizon': 256, 'vec.total_agents': 8192}
    objective = training_contract({'terminal_reward_mode': 'win_only', 'reward_scale': .5,
                                  'shaping_weight': .25, 'shaping_gamma': .999,
                                  'army_shaping_weight': .5, 'land_shaping_weight': .3}, overrides)
    manifest = write_asset(directory / 'asset', fabric=fabric(), factory_source_sha256='a' * 64,
                           model_sha256='b' * 64, abi_sha256='c' * 64, policy=policy,
                           sampler={'mode': 'structured_sample', 'move_temperature': .05, 'split_temperature': .15},
                           provenance=provenance or {
                               'operation': 'source_cleanup', 'reinforcement_learning_steps_added': 0,
                               'abi_proof_sha256': 'd' * 64, 'ancestors': {'unknown-research-master': 'e' * 64},
                               'opaque': {'method': 'supervised', 'updates': 256, 'training_data_sha256': 'f' * 64},
                           }, learner=state, training_seeds=[8842],
                           learner_configuration={'seed': 8842, 'overrides': overrides} if learner else None,
                           training_contract=objective if learner else None)
    return manifest, policy, state


def read(manifest):
    return load_asset(manifest, manifest_sha256=sha256(manifest.read_bytes()))


def test_source_cleanup_preserves_policy_learner_bytes_and_opaque_lineage(tmp_path):
    manifest, policy, state = author(tmp_path / 'source', learner=True)
    asset = read(manifest)
    assert asset.policy == policy.read_bytes()
    assert asset.learner == state.read_bytes()
    assert asset.metadata['provenance']['ancestors'] == {'unknown-research-master': 'e' * 64}
    assert asset.metadata['provenance']['opaque']['updates'] == 256
    assert asset.metadata['provenance']['reinforcement_learning_steps_added'] == 0
    assert set(p.name for p in manifest.parent.iterdir()) == {'asset.json', 'policy.bin', 'policy.bin.learner'}
    asset.verify_target(factory_source_sha256='a' * 64, model_sha256='b' * 64, abi_sha256='c' * 64)
    with pytest.raises(ValueError, match='target identity differs'):
        asset.verify_target(factory_source_sha256='f' * 64, model_sha256='b' * 64, abi_sha256='c' * 64)


def test_manifest_and_parameter_payload_tampering_rejected(tmp_path):
    manifest, _, _ = author(tmp_path / 'source')
    with pytest.raises(ValueError, match='manifest checksum'):
        load_asset(manifest, manifest_sha256='f' * 64)
    (manifest.parent / 'policy.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='finite float32'):
        read(manifest)


def test_source_cleanup_cannot_claim_rl_steps_or_change_architecture(tmp_path):
    manifest, _, _ = author(tmp_path / 'source')
    metadata = json.loads(manifest.read_bytes())
    metadata['provenance']['reinforcement_learning_steps_added'] = 8_388_608
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='zero RL steps'):
        read(manifest)
    metadata['provenance']['reinforcement_learning_steps_added'] = 0
    metadata['fabric']['options']['factorized_actions'] = False
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='architecture options'):
        read(manifest)


def test_current_restore_fields_are_mandatory_without_opaque_runtime_parsing(tmp_path):
    manifest, _, _ = author(tmp_path / 'source', learner=True)
    metadata = json.loads(manifest.read_bytes())
    metadata['learner_configuration'] = None
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='explicit actual configuration'):
        read(manifest)
    manifest, _, _ = author(tmp_path / 'policy-only')
    metadata = json.loads(manifest.read_bytes())
    metadata['training_seeds'] = [8842, 8842]
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='explicit unique'):
        read(manifest)


def test_restore_objective_excludes_distributions_but_binds_discounts():
    reward = {'terminal_reward_mode': 'win_only', 'reward_scale': .5, 'shaping_weight': .25,
              'shaping_gamma': .999, 'army_shaping_weight': .5, 'land_shaping_weight': .3}
    source = training_contract(dict(reward, frozen_bundle='/old', parallel_games=8192), {'train.gamma': .999})
    target = training_contract(dict(reward, frozen_bundle='/new', parallel_games=2048), {'train.gamma': .999})
    assert source == target
    with pytest.raises(ValueError, match='discounts must agree'):
        training_contract(reward, {'train.gamma': .99})


@pytest.mark.parametrize("scale", [0, 4])
def test_sampler_rejects_retired_log_gap_field(scale):
    from integrations.native_spatial_asset import validate_sampler

    with pytest.raises(ValueError, match="current explicit structured sampler"):
        validate_sampler(dict(mode="structured_sample", move_temperature=.05,
                              split_temperature=.15, log_gap_scale=scale))

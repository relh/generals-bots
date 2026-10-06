"""A test pool or misbound archive cannot enter PPO publication lineage."""

import hashlib
import json
from pathlib import Path

import pytest
from integrations.publish_policy_asset import curriculum_lineage


def test_actual_source_seeds_and_test_exclusion(tmp_path):
    pool = tmp_path / 'positions.npz'
    pool.write_bytes(b'authentic-placeholder-for-hash-boundary')
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.json'
    content = dict(schema='classic-midgame-positions-v1', split='source', teacher_labels_enabled=False,
                   positions_sha256=digest, root_training_seed=11, map_seeds=[13, 17])
    manifest.write_text(json.dumps(content))
    options = dict(coworld_position_pool=str(pool), coworld_position_pool_sha256=digest)
    seeds, stamp = curriculum_lineage(manifest, options)
    assert seeds == {11, 13, 17} and stamp == hashlib.sha256(manifest.read_bytes()).hexdigest()
    content['split'] = 'test'
    manifest.write_text(json.dumps(content))
    with pytest.raises(ValueError, match='source-only'):
        curriculum_lineage(manifest, options)
    content['split'] = 'source'
    manifest.write_text(json.dumps(content))
    pool.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='source-only'):
        curriculum_lineage(manifest, options)

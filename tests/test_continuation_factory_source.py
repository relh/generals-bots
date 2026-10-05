import hashlib
import json

import pytest

from integrations.classic_learner_continuation import verify_factory_source


def test_pinned_factory_is_read_only_and_generic_replacement_is_rejected(tmp_path):
    parent = tmp_path / 'parent'
    (parent / 'bundle').mkdir(parents=True)
    source = tmp_path / 'factory.py'
    source.write_bytes(b'pinned factory\n')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = parent / 'bundle/spatial-policy.json'
    manifest.write_text(json.dumps({'factory_source_sha256': expected}))
    original = manifest.read_bytes()
    assert verify_factory_source(parent, source) == expected
    assert manifest.read_bytes() == original
    source.write_bytes(b'generic factory with a different channel guard\n')
    with pytest.raises(ValueError, match='preserve its pinned source'):
        verify_factory_source(parent, source)
    assert manifest.read_bytes() == original


def test_missing_factory_identity_does_not_qualify(tmp_path):
    (tmp_path / 'bundle').mkdir()
    (tmp_path / 'bundle/spatial-policy.json').write_text('{}')
    source = tmp_path / 'factory.py'
    source.write_text('factory')
    with pytest.raises(ValueError, match='Continuation factory source differs'):
        verify_factory_source(tmp_path, source)

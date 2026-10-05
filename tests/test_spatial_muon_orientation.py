import copy

import pytest

from integrations.spatial_muon_orientation import (
    MARKER, canonical_dense_source, validate_build_mode, validate_geometry,
)


def configuration():
    return dict(
        factory="integrations.generals_fabric:two_stage_tied_local_action_policy",
        observation_size=7056, action_sizes=[3529],
        options=dict(height=21, width=21, channels=16, features_per_site=32,
                     global_features=32, context_radius=1.01, factorized_actions=False),
    )


@pytest.mark.parametrize("key,value", [
    ("channels", 11), ("features_per_site", 8), ("global_features", 64),
    ("context_radius", 2), ("factorized_actions", True),
])
def test_orientation_refuses_other_parameter_geometry(key, value):
    config = configuration()
    validate_geometry(config)
    config["options"][key] = value
    with pytest.raises(ValueError, match="sixteen-channel"):
        validate_geometry(config)


@pytest.mark.parametrize("key", ["teacher", "losses", "retrace", "routing"])
def test_orientation_refuses_extra_objectives(key):
    config = copy.deepcopy(configuration())
    config[key] = {"enabled": True}
    with pytest.raises(ValueError, match="plain PPO"):
        validate_geometry(config)


def test_kernel_patch_refuses_unknown_source():
    with pytest.raises(ValueError, match="pinned installed"):
        canonical_dense_source(b"float scale = 1;")


def test_runtime_refuses_mislabeled_cached_binary(tmp_path):
    binary = tmp_path / "puffer"
    binary.write_bytes(b"existing storage-oriented executable")
    validate_build_mode(tmp_path, "storage")
    with pytest.raises(ValueError, match="compiled executable"):
        validate_build_mode(tmp_path, "canonical")
    binary.write_bytes(MARKER.encode())
    with pytest.raises(ValueError, match="compiled executable"):
        validate_build_mode(tmp_path, "storage")

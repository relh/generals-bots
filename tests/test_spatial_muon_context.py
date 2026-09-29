from types import SimpleNamespace

import numpy as np
import pytest

from integrations.spatial_muon_context import (
    MARKER, OFFSET, context_source, load_gather, validate_model_gather,
)
from integrations.spatial_muon_orientation import validate_build_mode


def model():
    kernel = np.full((3, 3, 32, 32), -1, np.int32)
    neighbors = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    gather = load_gather().reshape(32, 32, 5)
    for output in range(32):
        for input_ in range(32):
            for index, (y, x) in enumerate(neighbors):
                kernel[y, x, input_, output] = OFFSET + gather[output, input_, index]
    return SimpleNamespace(context_kernel=kernel)


def test_gather_preserves_all_physical_weights_and_roundtrips():
    gather = validate_model_gather(model())
    physical = np.arange(5120, dtype=np.float32)
    matrix = physical[gather].reshape(32, 160)
    restored = np.empty_like(physical)
    restored[gather] = matrix.reshape(-1)
    np.testing.assert_array_equal(restored, physical)
    assert not np.array_equal(gather, np.arange(5120))


def test_changed_actual_sharing_order_is_refused():
    changed = model()
    changed.context_kernel[1, 1, 0, 0], changed.context_kernel[1, 1, 1, 0] = (
        changed.context_kernel[1, 1, 1, 0], changed.context_kernel[1, 1, 0, 0],
    )
    with pytest.raises(ValueError, match="sharing order"):
        validate_model_gather(changed)


def test_context_kernel_requires_pinned_dense_source():
    with pytest.raises(ValueError, match="pinned canonical"):
        context_source(b"unverified source")


def test_context_cached_binary_cannot_run_as_vector(tmp_path):
    (tmp_path / "puffer").write_bytes(MARKER.encode())
    with pytest.raises(ValueError, match="Convolution matrix mode"):
        validate_build_mode(tmp_path, "canonical")
    with pytest.raises(ValueError, match="requires canonical"):
        validate_build_mode(tmp_path, "storage", context_matrix=True)

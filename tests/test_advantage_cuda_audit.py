"""CPU checks of CUDA audit inputs and invocation; no GPU execution claim."""
from pathlib import Path

import numpy as np
import pytest

from integrations.audit_advantage_normalization import compiler_command, cuda_source, fixtures, reference


@pytest.mark.parametrize("horizon", [128, 256])
def test_reference_ignores_poison_and_standardizes_active_slots(horizon):
    for name, values in fixtures(horizon):
        expected = reference(values, horizon).reshape(-1, horizon)
        np.testing.assert_array_equal(expected[:, -1], 0)
        assert np.isfinite(expected).all()
        if name == "constant":
            np.testing.assert_array_equal(expected, 0)
        else:
            assert abs(expected[:, :-1].mean()) < 1e-12
            assert abs(expected[:, :-1].std(ddof=1) - 1) < 1e-8
        clean = values.copy().reshape(-1, horizon)
        clean[:, -1] = 0
        np.testing.assert_array_equal(reference(clean.ravel(), horizon), expected.ravel())


def test_compiles_production_header_with_native_fp32_declaration():
    precision = "#ifdef PRECISION_FLOAT\ntypedef float precision_t;\nconstexpr bool USE_BF16 = false;\n#endif"
    source = cuda_source("prefix\n" + precision + "\nsuffix")
    assert precision in source
    assert '#include "metta_advantage.cuh"' in source
    assert "metta_standardize_ppo_advantages<<<1,256,0,stream>>>(device,count,horizon);" in source
    assert "__global__" not in source  # The kernel comes only from the production header.
    assert compiler_command(Path('/build/src/metta_advantage.cuh'), Path('/audit.cu'), Path('/audit')) == [
        'nvcc', '-std=c++17', '-O2', '-DPRECISION_FLOAT', '-I', '/build/src', '/audit.cu', '-o', '/audit']
    with pytest.raises(ValueError, match="declaration changed"):
        cuda_source("#ifdef PRECISION_FLOAT\ntypedef double precision_t;\n#endif")

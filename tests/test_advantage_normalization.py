"""Bootstrap exclusion admission; CPU execution is not a CUDA reduction claim."""
import ast
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "integrations/puffer_coworld_frozen_transfer.py"
KERNEL = ROOT / "integrations/puffer_advantage_normalization.cuh"


@pytest.fixture(scope="module")
def cpu_kernel(tmp_path_factory):
    path = tmp_path_factory.mktemp("advantage")
    # Execute the actual kernel arithmetic with one logical thread. CUDA's
    # parallel reduction order and synchronization require separate GPU proof.
    source = '''#include <cmath>
#include <cstdio>
#include <vector>
#define __global__
#define __shared__
#define __syncthreads() ((void)0)
using precision_t = float;
struct { int x; } threadIdx{0}, blockDim{1};
float to_float(float x) { return x; }
float from_float(float x) { return x; }
''' + KERNEL.read_text() + '''
int main() {
  int count, horizon;
  if (scanf("%d %d", &count, &horizon) != 2) return 1;
  std::vector<float> values(count);
  for (float& v : values) if (scanf("%f", &v) != 1) return 2;
  metta_standardize_ppo_advantages(values.data(), count, horizon);
  for (float v : values) printf("%.9g\\n", v);
}
'''
    cpp = path / "kernel.cpp"
    cpp.write_text(source)
    executable = path / "kernel"
    subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(executable)], check=True)
    return executable


@pytest.mark.parametrize("rows,horizon", [(2, 2), (3, 4), (64, 128)])
@pytest.mark.parametrize("constant", [False, True])
def test_bootstrap_exclusion_matches_transition_reference(cpu_kernel, rows, horizon, constant):
    values = np.full((rows, horizon), 3.0, dtype=np.float32)
    if not constant:
        values[:] = np.random.default_rng(11).normal(2.0, 0.5, values.shape)
    # Poison bootstrap slots to prove neither moments nor outputs depend on them.
    values[:, -1] = np.nan
    result = subprocess.run([str(cpu_kernel)], input=f"{values.size} {horizon}\n" +
                            " ".join(map(str, values.flat)), text=True, capture_output=True, check=True)
    actual = np.fromstring(result.stdout, sep="\n").reshape(values.shape)
    active = values[:, :-1].astype(np.float64)
    expected = (active - active.mean()) / (active.std(ddof=1) + 1e-8)
    np.testing.assert_allclose(actual[:, :-1], expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(actual[:, -1], 0.0)
    assert np.isfinite(actual).all()


def test_installer_passes_actual_minibatch_horizon_and_pins_kernel(tmp_path):
    tree = ast.parse(TRAINER.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "install_advantage_normalization")
    scope = {"Path": Path, "hashlib": hashlib, "__file__": str(TRAINER)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER), "exec"), scope)
    (tmp_path / "src").mkdir()
    (tmp_path / "config").mkdir()
    target = tmp_path / "src/pufferl.cu"
    target.write_text('    float momentum;\n'
                      '        .momentum = puf_ini_get(ini, "train", "momentum"),\n'
                      'static void train_epoch_gpu(PuffeRL* pufferl, RolloutBuf src, int slot,\n'
                      '        ppo_loss_fwd_bwd(dec, p_logstd, graph,\n')
    (tmp_path / "config/default.ini").write_text("momentum = 0.95\n")
    scope[function.name](tmp_path)
    emitted = target.read_text()
    assert "graph.mb_advantages.data, count, Tmb);" in emitted
    assert "assert(Tmb > 1 && count % Tmb == 0 && count - count / Tmb > 1);" in emitted
    assert (tmp_path / "src/metta_advantage.cuh").read_bytes() == KERNEL.read_bytes()
    assert emitted.index("metta_standardize_ppo_advantages<<<") < emitted.index("ppo_loss_fwd_bwd(dec")
    with pytest.raises(ValueError, match="without native advantage normalization"):
        scope[function.name](tmp_path)

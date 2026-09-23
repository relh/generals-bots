#!/usr/bin/env bash
set -euo pipefail

export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH="/work/runtime/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import jax

if jax.devices()[0] not in jax.devices("cuda") or not jax.devices("cpu"):
    raise RuntimeError("Puffer training requires CUDA and CPU JAX backends")
print("JAX training device:", jax.devices("cuda")[0], flush=True)
PY

exec python -m metta_training.cli "$@"

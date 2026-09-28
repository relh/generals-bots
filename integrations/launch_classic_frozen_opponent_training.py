"""Run the pinned native trainer with the exact frozen-opponent transfer guard."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import jax

source = Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "862a6232b430dd804cf3fdcad62f70f24e778a6a5a233f6c1bdf2b25ea122e3f"
)
if jax.devices()[0] not in jax.devices("cuda") or not jax.devices("cpu"):
    raise RuntimeError("Puffer training requires CUDA and CPU JAX backends")
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runpy.run_module("metta_training.cli", run_name="__main__")

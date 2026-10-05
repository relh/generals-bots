"""Run the pinned Puffer CLI with the audited Classic policy transfer guard."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import jax

source = Path(__file__).with_name("puffer_selfplay_iter1_transfer-v1-20260927.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "e56cd5a8834af5231441404cd51366259d65a5009e0c000e7682df0eecda084f"
)
if jax.devices()[0] not in jax.devices("cuda") or not jax.devices("cpu"):
    raise RuntimeError("Puffer training requires CUDA and CPU JAX backends")
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runpy.run_module("metta_training.cli", run_name="__main__")

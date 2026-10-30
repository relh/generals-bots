"""Run the pinned Puffer CLI with the audited Classic policy transfer guard."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import jax

source = Path(__file__).with_name("puffer_selfplay_policy_transfer-v1-20260927.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "8f497fe300eade8653e7fc6b92b23cb0d850cc09d0fddf518de601603da1ea73"
)
if jax.devices()[0] not in jax.devices("cuda") or not jax.devices("cpu"):
    raise RuntimeError("Puffer training requires CUDA and CPU JAX backends")
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runpy.run_module("metta_training.cli", run_name="__main__")

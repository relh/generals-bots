"""Load verified control and frozen evaluator for zero-layer and recurrent native policies."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

root = Path(__file__).parent
source = root / "puffer-native-feedforward-20260927.py"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "25ec6d1b0f7c08012a58681ce44b5721143f6da10c78b6cf50e9dac01f7d9ccf"
)
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
evaluator = root / "evaluate_coworld_frozen_greedy-reward-20260927.py"
assert (
    hashlib.sha256(evaluator.read_bytes()).hexdigest()
    == "d9bf712f51556cda413a92aec356aa7cfe3346da6fe8dc1e6e43b48d85862b28"
)
runpy.run_path(str(evaluator), run_name="__main__")

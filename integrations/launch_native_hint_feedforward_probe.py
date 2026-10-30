"""Load verified run control while preserving archived environment fingerprints."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

source = Path(__file__).with_name("puffer-native-feedforward-20260927.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "25ec6d1b0f7c08012a58681ce44b5721143f6da10c78b6cf50e9dac01f7d9ccf"
)
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runner = Path(__file__).with_name("train_native_hint_initialized-feedforward-20260927.py")
assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
    "09ba27ac251f45f271d2089e624928cb99b3802c587b9f51ea709c6bd901cf4e"
)
runpy.run_path(str(runner), run_name="__main__")

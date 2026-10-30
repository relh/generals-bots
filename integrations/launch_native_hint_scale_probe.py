"""Load verified run control while preserving archived environment fingerprints."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

source = Path(__file__).with_name("puffer-native-initializer-20260927.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "0958d85285144647aea4d32fc60f08e3965d82d4cad83f7b184a58949cd63ec4"
)
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runner = Path(__file__).with_name("train_native_hint_initialized-scale-20260927.py")
assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
    "9d619be5267442f361e9c0f0a6f7a32dfb0f0596cae313cbe7197921d21faa55"
)
runpy.run_path(str(runner), run_name="__main__")

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
runner = Path(__file__).with_name("train_native_hint_initialized-lifecycle-20260927.py")
assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
    "89d57ff7f0af31646594f1f8836a081c3eb66bfeddb13f35ce75cad981fdad51"
)
runpy.run_path(str(runner), run_name="__main__")

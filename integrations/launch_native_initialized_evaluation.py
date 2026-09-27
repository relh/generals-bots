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
runpy.run_path(str(Path(__file__).with_name("evaluate_coworld_frozen_greedy-native-20260927.py")), run_name="__main__")

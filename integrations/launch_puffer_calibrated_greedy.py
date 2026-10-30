"""Load verified run-control changes without changing the archived environment files."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

source = Path(__file__).with_name("puffer-calibrated-transfer.py")
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    "37550f9dea166c0a7f8e6b165a2e52f12b41883f2b7599ac9f57aa5b12ca0eda"
)
spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
runpy.run_path(str(Path(__file__).with_name("evaluate_coworld_frozen_greedy.py")), run_name="__main__")

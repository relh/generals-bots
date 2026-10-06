"""The staged proof must match the runner's effective JSON contract."""

import json
import os
from pathlib import Path

import pytest

from integrations.source_mirror_probe import serializable_training_contract


def test_staged_probe_intent_contract_matches_runtime():
    root = Path(os.environ.get("GENERALS_SOURCE_MIRROR_INPUT", "/tmp/generals-source-mirror-probe-4cc-input"))
    if not (root / "probe-intent.json").exists():
        pytest.skip("Source-mirror staged input is not present on this machine")
    proof = json.loads((root / "probe-intent.json").read_text())
    build = json.loads((root / "build-config.json").read_text())
    run = json.loads((root / "config.json").read_text())
    assert serializable_training_contract(build, run) == proof["contract"]
    assert proof["contract"]["map_options"]["mountain_density_range"] == [0.24, 0.26]
    assert proof["contract"]["training_geometry"]["steps_per_epoch"] == 524_288

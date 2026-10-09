"""Fixed source assets and training configuration for controlled policy trials."""
import json
from pathlib import Path

CONFIG = Path(__file__).with_suffix("")
ASSETS = json.loads((CONFIG / "assets.json").read_text())

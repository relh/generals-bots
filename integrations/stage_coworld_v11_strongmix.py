"""Add a four-game strong opponent mix to the pinned v11 environment source."""

import hashlib
import shutil
import sys
from pathlib import Path


EXPECTED = "ae3f68b80edcee43dd2968eb438e2dbb6d7f156f4fae732079989f0b7c8f86cb"
OLD = '''        opponent_agents = (
            (RandomAgent(), ExpanderAgent(), HunterAgent()) if opponent == "mixed" else (opponent_types[opponent](),)
        )
'''
NEW = '''        opponent_agents = (
            (RandomAgent(), ExpanderAgent(), HunterAgent()) if opponent == "mixed"
            else (ExpanderHarvesterAgent(), ExpanderHarvesterAgent(), ExpanderHarvesterAgent(), SentinelAgent())
            if opponent == "strong_mixed" else (opponent_types[opponent](),)
        )
'''


def main() -> None:
    source, destination = map(Path, sys.argv[1:])
    original = source / "integrations" / "metta_puffer.py"
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("Unexpected v11 environment source")
    text = data.decode()
    if text.count(OLD) != 1:
        raise ValueError("Expected one opponent selection block")
    shutil.copytree(source, destination)
    patched = destination / "integrations" / "metta_puffer.py"
    patched.write_text(text.replace(OLD, NEW))
    print("source_sha256=" + EXPECTED)
    print("patched_sha256=" + hashlib.sha256(patched.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

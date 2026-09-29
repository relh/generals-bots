"""Reproduce the sixteen-channel variant of the exact archived spatial factory."""

import argparse
import hashlib
from pathlib import Path

from integrations.memoryless_optimization import FACTORY_SHA256, PUBLIC_SCALAR_FACTORY_SHA256


def extend_archived_factory(source):
    if hashlib.sha256(source).hexdigest() != FACTORY_SHA256:
        raise ValueError("Expected the verified archived spatial factory")
    old = b"and channels != 11:"
    if source.count(old) != 1:
        raise ValueError("Archived channel guard differs")
    extended = source.replace(old, b"and channels not in (11, 16):")
    if hashlib.sha256(extended).hexdigest() != PUBLIC_SCALAR_FACTORY_SHA256:
        raise ValueError("Extended spatial factory differs from its pinned identity")
    return extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extended = extend_archived_factory(args.source.read_bytes())
    with args.output.open("xb") as output:
        output.write(extended)


if __name__ == "__main__":
    main()

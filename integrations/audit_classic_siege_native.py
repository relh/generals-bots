"""Check a native siege build against retained public-input/action cases on CPU."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from integrations.classic_siege_native import SOURCE, ClassicSiegeBatch, compile_library


def audit(cases, library):
    opponent = ClassicSiegeBatch(library)
    with np.load(cases, allow_pickle=False) as archive:
        inputs = [archive[key] for key in ("dimensions", "turns", "grids", "memories")]
        expected = archive["expected"]
    start = time.perf_counter()
    actual, _ = opponent(*inputs)
    elapsed = time.perf_counter() - start
    if expected.shape != actual.shape or expected.dtype != np.int32:
        raise ValueError("Expected action shape or dtype differs")
    bad = np.flatnonzero(np.any(actual != expected, axis=1))
    return dict(
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        cases_sha256=hashlib.sha256(Path(cases).read_bytes()).hexdigest(),
        actions=len(actual), mismatches=len(bad), first_mismatch_indices=bad[:10].tolist(),
        elapsed_seconds=elapsed,
        scope="CPU opponent batch only; excludes JAX transfers, environment and learner",
        tie_break="row-major frontier; Python source uses set iteration",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.build:
        compile_library(args.library)
    report = audit(args.cases, args.library)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

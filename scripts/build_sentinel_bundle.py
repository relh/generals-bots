"""Build a deterministic, offline Sentinel submission zip without uploading it."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


VARIANTS = (
    "v2",
    "v3",
    "v3-memory",
    "v3-defense",
    "v3-disabled",
    "v4",
    "v4-adjacent",
    "v5",
    "v5-disabled",
    "v6",
    "v6-disabled",
)


def build(output, prewarm_cache=True, variant="v2"):
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    # Package initializers are intentionally minimal: inference does not need to
    # import the environment, GUI, trainers, or optional training dependencies.
    files = {
        name: b'"""Standalone Sentinel inference package."""\n'
        for name in ("generals/__init__.py", "generals/agents/__init__.py", "generals/core/__init__.py")
    }
    for name in (
        "generals/agents/sentinel_agent.py",
        "generals/agents/agent.py",
        "generals/core/action.py",
        "generals/core/observation.py",
    ):
        files[name] = (ROOT / name).read_bytes()
    files["main.py"] = (ROOT / "competition/agents/sentinel_python/main.py").read_bytes()
    if variant != "v2":
        name = "generals/agents/sentinel_v3_agent.py"
        files[name] = (ROOT / name).read_bytes()
    if variant.startswith("v4"):
        name = "generals/agents/sentinel_v4_agent.py"
        files[name] = (ROOT / name).read_bytes()
    if variant.startswith(("v5", "v6")):
        name = "generals/agents/sentinel_v5_agent.py"
        files[name] = (ROOT / name).read_bytes()
    if variant.startswith("v6"):
        name = "generals/agents/sentinel_v6_agent.py"
        files[name] = (ROOT / name).read_bytes()
    bootstrap = b"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
"""
    bootstrap += f"export SENTINEL_VARIANT={variant}\nexport SENTINEL_MODE=competition\n".encode()
    if prewarm_cache:
        bootstrap += b"""export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
"""
        files["warm_cache.py"] = (ROOT / "competition/agents/sentinel_python/warm_cache.py").read_bytes()
        files["build.sh"] = (
            bootstrap + b'export JAX_ENABLE_COMPILATION_CACHE=true\nexec python3 -u warm_cache.py "$@"\n'
        )
    files["run.sh"] = bootstrap + b"exec python3 -u main.py\n"
    files["LICENSE"] = (ROOT / "LICENSE").read_bytes()
    manifest = {
        "kind": "sentinel-standalone/1",
        "rules": "competition",
        "variant": variant,
        "entrypoint": "run.sh",
        "prewarm_cache": prewarm_cache,
        "prewarm_shapes": [[h, w] for h in range(18, 22) for w in range(18, 22)] if prewarm_cache else [],
        "python": "3.12.10",
        "preinstalled_dependencies": {"jax": "0.11.0", "numpy": "2.4.6"},
        "official_environment": "https://www.generals.bot/docs#environment",
        "source_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
        "notes": "No wheels, editable repository install, model weights, or network access required. "
        "Package initializers minimized; policy and adapter sources copied unchanged.",
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    assert len(files) <= 10000 and sum(map(len, files.values())) <= 512 * 1024**2
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(info, data)
    assert output.stat().st_size <= 50 * 1024**2
    return {
        "zip": str(output),
        "bytes": output.stat().st_size,
        "files": len(files),
        "unpacked_bytes": sum(map(len, files.values())),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "policy_sha256": manifest["source_sha256"][
            "generals/agents/sentinel_agent.py"
            if variant == "v2"
            else "generals/agents/sentinel_v6_agent.py"
            if variant.startswith("v6")
            else "generals/agents/sentinel_v5_agent.py"
            if variant.startswith("v5")
            else "generals/agents/sentinel_v4_agent.py"
            if variant.startswith("v4")
            else "generals/agents/sentinel_v3_agent.py"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="v2")
    parser.add_argument("--no-prewarm-cache", action="store_true", help="Omit build.sh and cache configuration")
    args = parser.parse_args()
    print(json.dumps(build(args.output, not args.no_prewarm_cache, args.variant), indent=2))


if __name__ == "__main__":
    main()

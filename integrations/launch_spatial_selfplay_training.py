"""Pinned Puffer launcher with a narrow, checked spatial-opponent transfer."""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import jax


def spatial_transfer(source, target, digest):
    if digest != "df706173df2c27fefe2279c8f1252d8eba376ad3a7a2858123d1c749afa44ddc":
        return False
    if source.model_sha256 != "2ca4d0da7ff313ae981f99728be0d1309fe679c8c0ff19fcc13a9a5d731a0c1e":
        return False
    if target.model_sha256 != source.model_sha256:
        return False
    before, after = source.config.python_environment, target.config.python_environment
    if before is None or after is None:
        return False
    if before.factory != "integrations.metta_puffer:BatchedGeneralsPufferEnvironment":
        return False
    if after.factory != "integrations.spatial_selfplay:SpatialFrozenOpponentPufferEnvironment":
        return False
    options = after.options.copy()
    if options.pop("frozen_bundle", None) != "/recovery/classic-spatial-local8-direct-eval-pilot-30342/bundle":
        return False
    return options == before.options


def main():
    source = Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != "4d18c06c59b4dad321bf61ba4d4aed552dc406159c96072e880cb518162ecea6":
        raise ValueError("Pinned Puffer trainer changed")
    if jax.devices()[0].platform != "gpu" or not jax.devices("cpu"):
        raise RuntimeError("Spatial self-play requires GPU and CPU JAX backends")
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.verified_classic_frozen_opponent_transfer
    module.verified_classic_frozen_opponent_transfer = lambda a, b, c: original(a, b, c) or spatial_transfer(a, b, c)
    runpy.run_module("metta_training.cli", run_name="__main__")


if __name__ == "__main__":
    main()

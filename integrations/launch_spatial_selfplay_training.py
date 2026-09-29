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
    if after.factory not in {
        "integrations.spatial_selfplay:SpatialFrozenOpponentPufferEnvironment",
        "integrations.spatial_selfplay:SpatialMixedFrozenOpponentPufferEnvironment",
    }:
        return False
    if before.spec != after.spec:
        return False
    options = after.options.copy()
    if options.pop("frozen_bundle", None) != "/recovery/classic-spatial-local8-direct-eval-pilot-30342/bundle":
        return False
    # A separate pilot may remove shaping and use the game's exact +/-1
    # capture outcome; no other reward or public-feature changes are admitted.
    if options.get("shaping_weight") == 0.0 and options.get("reward_scale") == 1.0:
        options["shaping_weight"] = before.options["shaping_weight"]
        options["reward_scale"] = before.options["reward_scale"]
    return options == before.options


def spatial_self_play_transfer(source, target, digest):
    """Admit the verified parent into two learning seats per physical game."""
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
    if after.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment":
        return False
    if before.spec != after.spec or before.spec.agents != 4096:
        return False
    if before.options.get("parallel_games") != 4096 or after.options.get("parallel_games") != 2048:
        return False
    if before.options.get("teacher") is not None or any(
        before.options.get(k) for k in ("supervise_teacher", "teacher_rollouts")
    ):
        return False
    options = after.options.copy()
    if "frozen_bundle" in options or options.get("shaping_weight") != 0 or options.get("reward_scale") != 1:
        return False
    options.update(parallel_games=before.options["parallel_games"],
                   shaping_weight=before.options["shaping_weight"],
                   reward_scale=before.options["reward_scale"])
    return options == before.options


def main():
    import os

    if os.environ.get("METTA_AUDIT_DEVICE_REWARDS") == "1":
        from integrations.environment_reward_audit import activate
        activate()
    source = Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != "3b7b12a77aee26bd7d611a7af0b6dcef2223cfeb074a83cff203ee823ae48b2f":
        raise ValueError("Pinned Puffer trainer changed")
    if jax.devices()[0].platform != "gpu" or not jax.devices("cpu"):
        raise RuntimeError("Spatial self-play requires GPU and CPU JAX backends")
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if os.environ.get("METTA_ALLOW_ENTROPY_COEFFICIENT_RESUME") == "1":
        from integrations.entropy_resume import (
            entropy_resume_overrides_compatible, entropy_resume_source,
        )
        module.entropy_resume_overrides_compatible = entropy_resume_overrides_compatible
        exec(compile(entropy_resume_source(source.read_text()), str(source), "exec"), module.__dict__)
    else:
        spec.loader.exec_module(module)
    orientation = os.environ.get("METTA_SPATIAL_MUON_DENSE_ORIENTATION", "storage")
    if orientation == "canonical":
        from integrations.spatial_muon_orientation import install_build_hook
        install_build_hook(module)
    elif orientation != "storage":
        raise ValueError("Spatial Muon orientation must be storage or canonical")
    context_mode = os.environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX", "0")
    if context_mode not in ("0", "1") or (context_mode == "1" and orientation != "canonical"):
        raise ValueError("Convolution matrix mode must be 0 or 1 and requires canonical dense scaling")
    if context_mode == "1":
        from integrations.spatial_muon_context import install_build_hook as install_context_build_hook
        install_context_build_hook(module)
    from integrations.spatial_muon_orientation import install_runtime_guard
    install_runtime_guard(module, orientation, context_matrix=context_mode == "1")
    original = module.verified_classic_frozen_opponent_transfer
    module.verified_classic_frozen_opponent_transfer = lambda a, b, c: (
        original(a, b, c) or spatial_transfer(a, b, c) or spatial_self_play_transfer(a, b, c)
    )
    runpy.run_module("metta_training.cli", run_name="__main__")


if __name__ == "__main__":
    main()

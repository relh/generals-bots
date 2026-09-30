"""Pinned Puffer launcher with a narrow, checked spatial-opponent transfer."""

import hashlib
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

def validate_training_geometry(argv=sys.argv, environ=os.environ):
    """Fail before GPU compilation if the verified wide trainer omits row flattening."""
    if len(argv) < 2 or argv[1] != "train" or "--config" not in argv:
        return
    config_path = Path(argv[argv.index("--config") + 1])
    overrides = json.loads(config_path.read_text()).get("overrides", {})
    if (environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX") == "1"
            and environ.get("METTA_DIRECT_SPATIAL_ROLLOUT") == "1"
            and int(overrides.get("vec.total_agents", 0)) >= 8192
            and int(overrides.get("train.horizon", 0)) >= 256
            and environ.get("METTA_MEMORYLESS_OPTIMIZATION") != "1"):
        raise ValueError(
            "Wide spatial PPO requires METTA_MEMORYLESS_OPTIMIZATION=1; "
            "without it JAX compiles the unflattened 8192-game device_core"
        )


def validate_sampling_gate(argv=sys.argv, environ=os.environ):
    """Require source-policy wins under the actual rollout sampler before PPO."""
    if len(argv) < 2 or argv[1] != "train" or "--config" not in argv or "--build" not in argv:
        return
    build = Path(argv[argv.index("--build") + 1]) / "build.json"
    if not build.exists():
        return  # The normal build loader reports its missing artifact.
    record = json.loads(build.read_text())
    environment = record["config"].get("python_environment") or {}
    options = environment.get("options") or {}
    if (environment.get("factory") not in {
            "integrations.spatial_selfplay:SpatialFrozenOpponentPufferEnvironment",
            "integrations.spatial_selfplay:SpatialMixedFrozenOpponentPufferEnvironment",
            "integrations.spatial_selfplay:SpatialPopulationOpponentPufferEnvironment",
        } or environment.get("spec", {}).get("action_sizes") != [3529]
            or options.get("terminal_reward_mode") != "win_only"):
        return
    training = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    initialization = training.get("initialize")
    if not initialization:
        return
    path = environ.get("METTA_SPATIAL_SAMPLING_GATE_REPORT")
    if not path:
        raise ValueError("Win-only flat-action transfer requires METTA_SPATIAL_SAMPLING_GATE_REPORT")
    report = json.loads(Path(path).read_text())
    source = initialization["sha256"]
    temperature = float(environ.get("METTA_SPATIAL_POLICY_TEMPERATURE", "1"))
    split_temperature = environ.get("METTA_SPATIAL_SPLIT_TEMPERATURE")
    split_temperature = float(split_temperature) if split_temperature is not None else None
    if (report.get("baseline_sha256") != source or report.get("candidate_sha256") != source
            or report.get("opponent_sha256") != source
            or report.get("baseline_action_selection") != "argmax"
            or report.get("candidate_action_selection") != "sample"
            or report.get("baseline_sampling_temperature") is not None
            or report.get("candidate_sampling_temperature") != temperature
            or report.get("candidate_split_sampling_temperature") != split_temperature
            or report.get("games", 0) < 256 or report.get("unique_initial_maps", 0) < 64):
        raise ValueError("Sampling gate must compare greedy and rollout-temperature play of the exact source policy")
    greedy_wins = report["baseline_wld"][0]
    sampled_wins = report["candidate_wld"][0]
    if greedy_wins < report["games"] / 4:
        raise ValueError(
            f"Greedy source wins only {greedy_wins}/{report['games']} in its self-match; "
            "verify the checkpoint and opponent before PPO"
        )
    minimum_wins = max(16, report["games"] / 4, greedy_wins / 2)
    if sampled_wins < minimum_wins:
        raise ValueError(
            f"Rollout sampling wins only {sampled_wins}/{report['games']} versus "
            f"greedy {greedy_wins}/{report['games']}; collect useful winning trajectories before PPO"
        )


def prepare_temporary_directory(environ=os.environ):
    """Create the configured compiler scratch directory before JAX or nvcc starts."""
    location = environ.get("TMPDIR")
    if location:
        Path(location).mkdir(parents=True, exist_ok=True)


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
    validate_training_geometry()
    validate_sampling_gate()
    prepare_temporary_directory()

    import jax

    if os.environ.get("METTA_AUDIT_DEVICE_REWARDS") == "1":
        from integrations.environment_reward_audit import activate
        activate()
    source = Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
    if hashlib.sha256(source.read_bytes()).hexdigest() != "9e09bbd9b541f3e1522195d07083e9be972d0a7ba6d187a8b21ff8c1e522d63e":
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

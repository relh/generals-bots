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
    if (
        environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX") == "1"
        and environ.get("METTA_DIRECT_SPATIAL_ROLLOUT") == "1"
        and int(overrides.get("vec.total_agents", 0)) >= 8192
        and int(overrides.get("train.horizon", 0)) >= 256
        and environ.get("METTA_MEMORYLESS_OPTIMIZATION") != "1"
    ):
        raise ValueError(
            "Wide spatial PPO requires METTA_MEMORYLESS_OPTIMIZATION=1; "
            "without it JAX compiles the unflattened 8192-game device_core"
        )


def rollout_sampler_settings(environ=os.environ):
    """One explicit actor contract shared by source qualification and PPO."""
    import math

    from integrations.spatial_action_sampling import validate_full_action_temperature

    if "METTA_SPATIAL_LOG_GAP_SCALE" in environ:
        raise ValueError("Unsupported retired sampler environment: METTA_SPATIAL_LOG_GAP_SCALE")
    settings = dict(
        mode="structured_sample",
        move_temperature=float(environ.get("METTA_SPATIAL_POLICY_TEMPERATURE", "1")),
        split_temperature=float(environ.get("METTA_SPATIAL_SPLIT_TEMPERATURE", "1")),
        full_action_temperature=validate_full_action_temperature(
            float(environ.get("METTA_SPATIAL_FULL_ACTION_TEMPERATURE", "1"))
        ),
        route_half_weight=float(environ.get("METTA_SPATIAL_ROUTE_HALF_WEIGHT", "0")),
        neutral_route_bias=float(environ.get("METTA_SPATIAL_NEUTRAL_ROUTE_BIAS", "0")),
        weak_owned_route_penalty=float(environ.get("METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY", "0")),
        doomed_attack_route_penalty=float(environ.get("METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY", "0")),
    )
    early = environ.get("METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE")
    turns = environ.get("METTA_SPATIAL_EARLY_ROUTE_TURNS")
    if (early is None) != (turns is None):
        raise ValueError("Early route sampling requires temperature and turns together")
    settings.update(
        early_route_temperature=float(early) if early is not None else None,
        early_route_turns=int(turns) if turns is not None else None,
    )
    if (
        any(not math.isfinite(v) for v in settings.values() if isinstance(v, (float, int)))
        or settings["move_temperature"] <= 0
        or settings["split_temperature"] <= 0
        or not 0 <= settings["route_half_weight"] <= 1
        or any(
            settings[k] < 0 for k in ("neutral_route_bias", "weak_owned_route_penalty", "doomed_attack_route_penalty")
        )
        or (early is not None and (settings["early_route_temperature"] <= 0 or settings["early_route_turns"] <= 0))
    ):
        raise ValueError("Rollout sampler settings must be finite and valid")
    return settings


def source_sampling_gate_report(match):
    """Build source viability evidence from one real balanced self-match."""
    import numpy as np

    match = Path(match)
    record = json.loads((match / "evaluation.json").read_text())
    if record.get("schema") != "generals-frozen-match-v2":
        raise ValueError("Source gate requires the current frozen-match report schema")
    if "log_gap_scale" in record or "log_gap_scale" in record.get("opponent_action_parameters", {}):
        raise ValueError("Source report contains retired sampler settings")
    if (
        not record["held_out"]
        or record["smoke_cpu"]
        or not record["coworld_classic_rules"]
        or record["episode_limit"] != 2000
        or record["action_selection"] != "sample"
        or record["opponent_action_selection"] != "structured_sample"
    ):
        raise ValueError("Source gate requires held-out official GPU Classic sampled self-play")
    if record["half_logit_bias"] != 0:
        raise ValueError("Source gate requires the actual PPO sampler without evaluation-only biases")
    hashes = np.load(match / "initial_state_sha256.npy", allow_pickle=False)
    sides = np.load(match / "initial_sides.npy", allow_pickle=False)
    outcomes = np.load(match / "outcomes.npy", allow_pickle=False)
    games = record["games"]
    if (
        hashes.shape != (games,)
        or sides.shape != (games,)
        or outcomes.shape != (games,)
        or not np.isin(outcomes, (-1, 0, 1)).all()
        or not np.isin(sides, (0, 1)).all()
    ):
        raise ValueError("Source match saved state, seat or outcome arrays differ")
    wld = [int((outcomes == value).sum()) for value in (1, -1, 0)]
    if wld != [record[k] for k in ("wins", "losses", "draws")]:
        raise ValueError("Source match outcomes disagree with its evaluation record")
    sampler = dict(
        mode="structured_sample",
        move_temperature=record["sampling_temperature"],
        split_temperature=record["split_sampling_temperature"],
    )
    for k in (
        "early_route_temperature",
        "early_route_turns",
        "route_half_weight",
        "full_action_temperature",
        "neutral_route_bias",
        "weak_owned_route_penalty",
        "doomed_attack_route_penalty",
    ):
        sampler[k] = record[k]
    opponent = dict(mode=record["opponent_action_selection"], **record["opponent_action_parameters"])
    return dict(
        gate_mode="same_sampler_source",
        source_sha256=record["checkpoint_sha256"],
        opponent_sha256=record["opponent_sha256"],
        sampler=sampler,
        opponent_sampler=opponent,
        games=games,
        unique_initial_maps=len(np.unique(hashes)),
        initial_states_sha256=hashlib.sha256(hashes.tobytes()).hexdigest(),
        seat_counts={str(side): int((sides == side).sum()) for side in (0, 1)},
        wld=wld,
        held_out=True,
        smoke_cpu=False,
        coworld_classic_rules=True,
        episode_limit=2000,
        match_seed=record["seed"],
        sample_seed=record["sample_seed"],
    )


def validate_sampling_gate(argv=sys.argv, environ=os.environ):
    """Require the exact source under its actual sampler before any PPO updates."""
    if len(argv) < 2 or argv[1] != "train" or "--config" not in argv or "--build" not in argv:
        return
    build = Path(argv[argv.index("--build") + 1]) / "build.json"
    if not build.exists():
        return  # The normal build loader reports its missing artifact.
    environment = json.loads(build.read_text())["config"].get("python_environment") or {}
    options = environment.get("options") or {}
    training = json.loads(Path(argv[argv.index("--config") + 1]).read_text())
    initialization = training.get("initialize")
    if not initialization:
        return
    if environment.get("spec", {}).get("action_sizes") != [3529] or options.get("terminal_reward_mode") != "win_only":
        raise ValueError("Source qualification requires the current flat Classic win objective")
    path = environ.get("METTA_SPATIAL_SAMPLING_GATE_REPORT")
    if not path:
        raise ValueError("Policy initialization requires METTA_SPATIAL_SAMPLING_GATE_REPORT")
    report = json.loads(Path(path).read_text())
    settings = rollout_sampler_settings(environ)
    if settings["early_route_temperature"] is not None and environment.get("spec", {}).get("observation_size") != 7056:
        raise ValueError("Early route schedule requires full public scalar turn observations")
    from integrations.native_spatial_asset import load_asset

    source_asset = load_asset(Path(initialization["asset"]), manifest_sha256=initialization["manifest_sha256"])
    source = source_asset.metadata["policy_sha256"]
    games = report.get("games", 0)
    if (
        report.get("gate_mode") != "same_sampler_source"
        or report.get("source_sha256") != source
        or report.get("opponent_sha256") != source
        or report.get("sampler") != settings
        or report.get("opponent_sampler") != settings
        or not report.get("held_out")
        or report.get("smoke_cpu")
        or report.get("coworld_classic_rules") is not True
        or report.get("episode_limit") != 2000
        or isinstance(games, bool)
        or not isinstance(games, int)
        or games < 512
        or games % 2
        or report.get("unique_initial_maps", 0) < 256
        or report.get("seat_counts") != {"0": games // 2, "1": games // 2}
    ):
        raise ValueError(
            "Source gate must verify the exact policy and both actors' intended rollout sampler on balanced fresh maps"
        )
    wld = report.get("wld", [])
    if (
        len(wld) != 3
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in wld)
        or sum(wld) != games
        or wld[0] < games / 4
    ):
        raise ValueError("Rollout sampling wins too few source self-matches for useful PPO trajectories")


def prepare_temporary_directory(environ=os.environ):
    """Create the configured compiler scratch directory before JAX or nvcc starts."""
    location = environ.get("TMPDIR")
    if location:
        Path(location).mkdir(parents=True, exist_ok=True)


def configure_offline_puffer(module, environ=os.environ):
    # Portable Slurm inputs carry the pinned upstream Git objects through S3.
    # Keep the trainer's exact revision check; never fetch code over the network
    # from a compute allocation when an audited local source is requested.
    puffer_source = environ.get("METTA_PUFFER_SOURCE_REPOSITORY")
    if puffer_source:
        repository = Path(puffer_source).resolve(strict=True)
        if not (repository / "objects").is_dir() or not (repository / "HEAD").is_file():
            raise ValueError("Portable Puffer source must be a local bare Git repository")
        module.PUFFER_REPOSITORY = str(repository)
        import shutil

        raylib = Path(environ["METTA_PUFFER_RAYLIB_DIRECTORY"]).resolve(strict=True)
        if not (raylib / "lib/libraylib.a").is_file() or not (raylib / "include/raylib.h").is_file():
            raise ValueError("Portable Puffer source requires its prepackaged raylib dependency")
        install_environment = module.install_environment

        def install_portable_environment(source, *args, **kwargs):
            shutil.copytree(raylib, source / "raylib-5.5_linux_amd64")
            return install_environment(source, *args, **kwargs)

        module.install_environment = install_portable_environment


def verify_runtime_bootstrap(environ=os.environ):
    if environ.get("METTA_MEMORYLESS_OPTIMIZATION") != "1":
        return
    if environ.get("METTA_SPATIAL_BOOTSTRAP_PID") != str(os.getpid()):
        raise RuntimeError("Spatial training requires the bootstrap in this interpreter, not only its environment flag")
    from integrations.activate_memoryless_optimization import AdapterFinder

    if not any(isinstance(finder, AdapterFinder) for finder in sys.meta_path):
        raise RuntimeError("Direct spatial adapter import hook is not active")


# Reviewed trainer supports verified current-model transfer and learner resume.
# Keep this identity explicit: updating trainer code requires reviewing its ABI
# and updating this binding together; an archive manifest alone is insufficient.
PINNED_TRAINER_SHA256 = "57d4118ce3a2bc993eeb6655eb9b7115cb9d0bb9472ca62186d95819028ee4f3"


def verify_trainer_source(source=None):
    source = Path(source) if source is not None else Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != PINNED_TRAINER_SHA256:
        raise ValueError(f"Pinned Puffer trainer changed: expected {PINNED_TRAINER_SHA256}, found {digest}")
    return source


def load_pinned_trainer():
    """Load the identical checked trainer and runtime hooks on CPU and GPU."""
    source = verify_trainer_source()
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    configure_offline_puffer(module)
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
    return module


def cpu_preflight(argv):
    """Exercise launch binding, native build guard and resume without GPU work."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Preflight has the same rollout geometry requirements as training.
    validate_training_geometry([sys.argv[0], "train", "--config", str(args.config)])
    from integrations.classic_contract import validate_training_contract

    run_config = json.loads(args.config.read_text())
    contract = validate_training_contract(json.loads((args.build / "build.json").read_text())["config"], run_config)
    module = load_pinned_trainer()
    prepared = module.prepare_run(args.build, args.output, module.RunConfig.model_validate(run_config))
    receipt = dict(
        trainer_sha256=PINNED_TRAINER_SHA256,
        puffer_revision=module.PUFFER_REVISION,
        contract=contract,
        batch_steps=prepared.batch_steps,
        initialization_verified=prepared.initialization is not None,
        scope="CPU launcher, build and initialization; sampling and GPU execution require runtime qualification",
    )
    print("SPATIAL_LAUNCH_CPU_READY " + json.dumps(receipt), flush=True)
    return receipt


def main():
    verify_runtime_bootstrap()
    prepare_temporary_directory()
    if len(sys.argv) > 1 and sys.argv[1] == "preflight":
        cpu_preflight(sys.argv[2:])
        return
    validate_training_geometry()
    validate_sampling_gate()
    # Check source before device discovery so an obsolete binding is caught
    # even on machines without a GPU.
    verify_trainer_source()
    import jax

    if os.environ.get("METTA_AUDIT_DEVICE_REWARDS") == "1":
        from integrations.environment_reward_audit import activate

        activate()
    if jax.devices()[0].platform != "gpu" or not jax.devices("cpu"):
        raise RuntimeError("Spatial self-play requires GPU and CPU JAX backends")
    load_pinned_trainer()
    runpy.run_module("metta_training.cli", run_name="__main__")


if __name__ == "__main__":
    main()

"""Build and execute a pinned upstream PufferLib engine."""

import hashlib
import os
import subprocess
import sys
from configparser import ConfigParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from metta_training.config import Configuration
from metta_training.environment import NativeEnvironmentCheckpoint, PythonEnvironmentConfig
from metta_training.game import Record
from metta_training.learner import LearnerCheckpoint, LearnerCheckpointIdentity
from metta_training.lifecycle import RunIdentity, TrainingArtifact
from metta_training.model_config import (
    FabricConfig,
    ObjectiveSettings,
)
from metta_training.native_build import (
    cuda_runtime_environment,
    environment_fingerprint,
    environment_runtime,
    fabric_fingerprint,
    fabric_runtime,
    install_driver,
    install_environment,
    install_fabric,
)
from metta_training.telemetry import RunMonitor, RunSnapshot

PUFFER_REPOSITORY = "https://github.com/PufferAI/PufferLib.git"
PUFFER_REVISION = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"


def install_advantage_normalization(source: Path) -> None:
    """Add Metta's opt-in actor normalizer after GAE/retrace, preserving returns."""
    path = source / "src/pufferl.cu"
    text = path.read_text()
    kernel_bytes = Path(__file__).with_name("puffer_advantage_normalization.cuh").read_bytes()
    if hashlib.sha256(kernel_bytes).hexdigest() != "f86375c1190da5abe28bdc74a95b10cdfd51011bb59a53ad1bea55e96cb87cef":
        raise ValueError("Advantage normalizer differs from the pinned Metta implementation")
    kernel = kernel_bytes.decode()
    replacements = (
        ("    float momentum;\n", "    float momentum;\n    bool norm_adv;\n"),
        ('        .momentum = puf_ini_get(ini, "train", "momentum"),',
         '        .momentum = puf_ini_get(ini, "train", "momentum"),\n'
         '        .norm_adv = puf_ini_get(ini, "train", "norm_adv") != 0,'),
        ("static void train_epoch_gpu(PuffeRL* pufferl, RolloutBuf src, int slot,",
         '#include "metta_advantage.cuh"\n\n'
         "static void train_epoch_gpu(PuffeRL* pufferl, RolloutBuf src, int slot,"),
        ("        ppo_loss_fwd_bwd(dec, p_logstd, graph,",
         "        if (hypers->norm_adv) {\n"
         "            int count = (int)numel(graph.mb_advantages.shape);\n"
         "            assert(Tmb > 1 && count % Tmb == 0 && count - count / Tmb > 1);\n"
         "            metta_standardize_ppo_advantages<<<1, 256, 0, stream>>>(\n"
         "                graph.mb_advantages.data, count, Tmb);\n"
         "        }\n"
         "        ppo_loss_fwd_bwd(dec, p_logstd, graph,"),
    )
    ini_path = source / "config/default.ini"
    ini = ini_path.read_text()
    if "norm_adv" in text or "norm_adv" in ini:
        raise ValueError("Expected the pinned trainer without native advantage normalization")
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Advantage normalization anchor changed: {old}")
        text = text.replace(old, new, 1)
    if ini.count("momentum = 0.95\n") != 1:
        raise ValueError("Puffer advantage default anchor changed")
    path.write_text(text)
    (source / "src/metta_advantage.cuh").write_text(kernel)
    ini_path.write_text(ini.replace("momentum = 0.95\n", "momentum = 0.95\nnorm_adv = 0\n", 1))


def install_minibatch_rotation(source: Path) -> None:
    """Rotate contiguous rollout blocks so fractional replay cannot starve rows."""
    path = source / "src/pufferl.cu"
    text = path.read_text()
    old = "        int dest_off = (mb * mb_segs) % n_rows;"
    new = """        // Host epoch offsets require graph-disabled execution.
        assert(!hypers->cudagraphs && "Minibatch rotation requires CUDA graphs disabled");
        assert(Nmb == mb_segs && n_rows % mb_segs == 0 && total_minibatches > 0);
        int blocks = n_rows / mb_segs;
        int start_block = (int)(((int64_t)pufferl->epoch * total_minibatches) % blocks);
        int dest_off = ((start_block + mb) % blocks) * mb_segs;
        if (mb == 0) {
            printf("MINIBATCH_ROTATION epoch=%ld start_block=%d total_minibatches=%d total_blocks=%d rows_per_block=%d rule=epoch_times_updates_mod_blocks\\n",
                pufferl->epoch, start_block, total_minibatches, blocks, mb_segs);
            fflush(stdout);
        }"""
    if text.count(old) != 1:
        raise ValueError("Pinned Puffer minibatch selection anchor changed")
    path.write_text(text.replace(old, new, 1))


def validate_minibatch_rotation(settings: ConfigParser) -> None:
    """Validate the exact contiguous-row schedule before native training."""
    if settings.getint("base", "cudagraphs") != -1:
        raise ValueError("Minibatch rotation requires base.cudagraphs=-1")
    agents = settings.getint("vec", "total_agents")
    horizon = settings.getint("train", "horizon")
    minibatch = settings.getint("train", "minibatch_size")
    replay = settings.getfloat("train", "replay_ratio")
    if (agents <= 0 or horizon <= 0 or minibatch <= 0
            or minibatch % horizon or (agents * horizon) % minibatch
            or not 0 < replay < float("inf")
            or int(replay * agents * horizon / minibatch) < 1):
        raise ValueError("Minibatch rotation requires whole contiguous row blocks and at least one update")


class BuildConfig(Configuration):
    environment: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    mode: Literal["train", "eval"] = "train"
    environment_backend: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32", "bfloat16"] = "float32"
    fabric: FabricConfig | None = None
    python_environment: PythonEnvironmentConfig | None = None

    @model_validator(mode="after")
    def validate_backend(self) -> "BuildConfig":
        if self.mode == "eval" and self.environment_backend == "cuda":
            raise ValueError("CPU evaluation requires a CPU environment")
        device_python = bool(self.python_environment and self.python_environment.device_resident)
        if self.fabric and (
            self.mode != "train"
            or self.environment_backend not in ({"cpu", "cuda"} if device_python else {"cpu"})
            or self.precision != "float32"
        ):
            raise ValueError("Fabric requires the float32 CUDA trainer with CPU environments")
        if self.fabric and self.fabric.teacher and not self.python_environment:
            raise ValueError("Teacher supervision requires a Python environment with explicit targets")
        if self.fabric and self.fabric.routing and not self.python_environment:
            raise ValueError("Trajectory routing requires a Python environment with explicit assignments")
        if self.fabric and self.fabric.replay_metadata_size and not self.python_environment:
            raise ValueError("Replay metadata requires a Python environment")
        if self.python_environment:
            if device_python:
                if self.environment_backend != "cuda" or self.mode != "train":
                    raise ValueError("Device-resident Python environments require CUDA training")
                if self.precision != "float32":
                    raise ValueError("Device-resident Python environments require float32 training")
                if self.python_environment.workers != "inline":
                    raise ValueError("Device-resident Python environments run in the native trainer")
                if self.python_environment.spec.replay_metadata_size or self.python_environment.spec.routing:
                    raise ValueError("Device-resident Python replay and routing transport is not implemented")
            if self.python_environment.spec.replay_metadata_size and not self.fabric:
                raise ValueError("Replay metadata requires Fabric; native models would consume privileged data")
            if self.fabric and self.fabric.replay_metadata_size != self.python_environment.spec.replay_metadata_size:
                raise ValueError("Model and environment must agree on replay metadata transport")
            if self.python_environment.spec.teacher and not self.fabric:
                raise ValueError("Teacher targets require Fabric; native models would consume privileged data")
            if self.fabric and bool(self.fabric.teacher) != self.python_environment.spec.teacher:
                raise ValueError("Model and environment must agree on teacher target transport")
            if self.python_environment.spec.routing and not self.fabric:
                raise ValueError("Routing assignments require Fabric; native models would consume privileged data")
            if self.fabric and bool(self.fabric.routing) != self.python_environment.spec.routing:
                raise ValueError("Model and environment must agree on routing assignment transport")
            if (
                self.fabric
                and self.fabric.routing
                and (self.fabric.routing.agents_per_game != self.python_environment.spec.agents)
            ):
                raise ValueError("Routing agent groups must match the environment's agent count")
            if not device_python and (self.mode != "train" or self.environment_backend != "cpu"):
                raise ValueError("Python environments require the native CUDA trainer with CPU environments")
            if self.fabric and (
                self.fabric.observation_size != self.python_environment.spec.observation_size
                or self.fabric.action_sizes != self.python_environment.spec.action_sizes
            ):
                raise ValueError("Fabric dimensions must match the environment specification")
        return self


class BuildManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: Literal["6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"] = PUFFER_REVISION
    config: BuildConfig
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    model_state_words: int = Field(default=0, ge=0)
    native_admission_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    environment_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")

    @model_validator(mode="after")
    def validate_model_identity(self) -> "BuildManifest":
        if bool(self.config.python_environment) != bool(self.environment_sha256):
            raise ValueError("Python environments require an implementation fingerprint")
        if self.config.fabric:
            from integrations import native_startup_admission
            if self.native_admission_sha256 != native_startup_admission.digest(native_startup_admission.__file__):
                raise ValueError("Native startup admission changed; rebuild the executable")
            if not self.model_sha256 or self.model_state_words != 0:
                raise ValueError("Spatial builds require a model fingerprint and zero external state")
        elif self.model_sha256 or self.model_state_words or self.native_admission_sha256:
            raise ValueError("Native models cannot declare Fabric metadata")
        return self


class CheckpointInitialization(Configuration):
    """One verified native policy asset, with explicit learner restoration."""

    asset: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    restore_learner: bool


class RunConfig(Configuration):
    total_timesteps: int = Field(gt=0)
    seed: int = Field(default=73, ge=0)
    initialize: CheckpointInitialization | None = None
    overrides: dict[str, str | int | FiniteFloat] = Field(default_factory=dict)

    @classmethod
    def synchronous(
        cls,
        total_timesteps: int,
        total_agents: int,
        horizon: int = 32,
        *,
        overrides: dict[str, str | int | float] | None = None,
    ) -> "RunConfig":
        return cls(
            total_timesteps=total_timesteps,
            overrides={
                "vec.total_agents": total_agents,
                "vec.num_buffers": 1,
                "vec.num_threads": 1,
                "train.horizon": horizon,
                "train.minibatch_size": total_agents * horizon,
                "train.replay_ratio": 1,
                "base.eval_episodes": 0,
                "sweep.downsample": 1,
            }
            | (overrides or {}),
        )

    @model_validator(mode="after")
    def validate_overrides(self) -> "RunConfig":
        reserved = {
            "train.total_timesteps",
            "base.seed",
            "base.checkpoint_dir",
            "base.log_dir",
            "base.run_id",
            "base.result_fd",
            "base.load_model_path",
            "base.load_enemy_model_path",
            "base.env_name",
        }
        for key, value in self.overrides.items():
            if key in reserved:
                raise ValueError(f"Run owns {key}; it cannot be overridden")
            if len(key.split(".")) < 2 or any(not part.replace("_", "").isalnum() for part in key.split(".")):
                raise ValueError(f"Expected a dotted upstream configuration key: {key}")
            if isinstance(value, str) and any(character in value for character in "\r\n\0"):
                raise ValueError(f"Configuration value contains a control character: {key}")
        return self


class TrainingRecord(Record):
    build: BuildManifest
    config: RunConfig


class InitializationRecord(Record):
    asset: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_seeds: list[int] = Field(min_length=1)
    restore_learner: bool


def training_lineage_seeds(run: Path, record: TrainingRecord) -> set[int]:
    seeds = {record.config.seed}
    if record.config.initialize:
        initialization = InitializationRecord.model_validate_json((run / "initialization.json").read_text())
        seeds.update(initialization.training_seeds)
    return seeds


class TrainingResult(Record):
    checkpoints: list[Path]
    final_checkpoint: Path
    revision: str
    trained_timesteps: int = Field(gt=0)
    metrics: Path


class PreparedRun(Record):
    build: Path
    output: Path
    name: str
    manifest: BuildManifest
    config: RunConfig
    overrides: dict[str, str | int | float]
    environment: dict[str, str] = Field(repr=False)
    expected_step: int
    world_size: int
    batch_steps: int
    environment_count: int
    initial_parameters: bytes = Field(default=b"", repr=False)
    initial_learner: bytes = Field(default=b"", repr=False)
    initialization: InitializationRecord | None = None


def build_puffer(output: Path, config: BuildConfig) -> BuildManifest:
    """Build in an exclusive directory; never modify another checkout or build."""
    if config.fabric and (os.environ.get("METTA_SPATIAL_MUON_DENSE_ORIENTATION") != "canonical"
                          or os.environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX") != "1"):
        raise ValueError("Current spatial builds require canonical dense and context-matrix Muon hooks")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", PUFFER_REPOSITORY, PUFFER_REVISION], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PUFFER_REVISION:
        raise RuntimeError(f"Puffer source mismatch: {revision}")
    if config.python_environment:
        install_environment(source, config.environment, config.python_environment)
        if config.python_environment.device_resident:
            from integrations.puffer_device_output import install_output_fence
            install_output_fence(source)
    suffix = "cu" if config.environment_backend == "cuda" else "h"
    if not (source / "ocean" / config.environment / f"{config.environment}.{suffix}").is_file():
        raise ValueError(f"Upstream has no {config.environment_backend} environment for {config.environment}")
    command = ["bash", "build.sh", config.environment, str(output / "puffer")]
    if config.mode == "eval":
        command.append("--cpu")
    if config.environment_backend == "cuda":
        command.append("--cu")
    if config.precision == "float32":
        command.append("--float")
    state_words = 0
    if config.fabric:
        state_words = install_fabric(source, config.fabric)
    environment = install_driver(
        source,
        cooperative_wait=bool(config.python_environment and not config.python_environment.device_resident),
        device_python_env=bool(config.python_environment and config.python_environment.device_resident),
    )
    install_advantage_normalization(source)
    install_minibatch_rotation(source)
    if config.fabric:
        from integrations.puffer_stateless_spatial import install_stateless_spatial
        install_stateless_spatial(source)
        from integrations.native_startup_admission import install as install_startup_admission
        install_startup_admission(source)
        from integrations.puffer_rollout_memory import install as install_rollout_memory
        install_rollout_memory(source)
        from integrations.spatial_muon_orientation import finalize_build_receipt
        finalize_build_receipt(source, config.fabric)
    model_digest = fabric_fingerprint(config.fabric) if config.fabric else ""
    with (output / "build.log").open("x") as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    manifest = BuildManifest(
        config=config,
        binary_sha256=hashlib.sha256((output / "puffer").read_bytes()).hexdigest(),
        model_sha256=model_digest,
        model_state_words=state_words,
        native_admission_sha256=hashlib.sha256(Path(__file__).with_name("native_startup_admission.py").read_bytes()).hexdigest() if config.fabric else "",
        environment_sha256=environment_fingerprint(config.python_environment) if config.python_environment else "",
    )
    (output / "build.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest


def prepare_run(build: Path, output: Path, config: RunConfig, *, name: str | None = None) -> PreparedRun:
    """Validate native configuration before allocating a run or a child trial."""
    build = build.resolve()
    output = output.resolve()
    manifest = BuildManifest.model_validate_json((build / "build.json").read_text())
    if manifest.config.mode != "train":
        raise ValueError("CPU evaluation binaries cannot train")
    if manifest.config.fabric and manifest.config.fabric.platform != "cuda":
        raise ValueError("Native Puffer training requires a CUDA Fabric model")
    binary = build / "puffer"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest.binary_sha256:
        raise ValueError("Puffer binary differs from its build manifest")
    settings = ConfigParser(interpolation=None)
    for path in (build / "source/config/default.ini", build / f"source/config/{manifest.config.environment}.ini"):
        with path.open() as source_config:
            settings.read_file(source_config)
    overrides = dict(config.overrides)
    environment = dict(os.environ)
    if manifest.config.fabric:
        owned, environment = fabric_runtime(manifest.config.fabric, manifest.model_sha256, manifest.model_state_words)
        if manifest.config.python_environment and manifest.config.python_environment.device_resident:
            owned["base.cudagraphs"] = -1
        environment["METTA_MODEL_METRICS"] = str(output / "teacher-metrics.json")
        for key, value in owned.items():
            if key in overrides and overrides[key] != value:
                raise ValueError(f"Fabric owns {key}={value}")
        overrides.update(owned)
        objective_settings = {key: value for key, value in overrides.items() if key.startswith("objective.")}
        if not objective_settings.keys() <= manifest.config.fabric.objective_defaults.keys():
            raise ValueError("Runtime objective settings must name declared numeric loss or optimizer options")
        environment["METTA_OBJECTIVE_SETTINGS"] = ObjectiveSettings.model_validate(objective_settings).model_dump_json()
        overrides = {key: value for key, value in overrides.items() if key not in objective_settings}
    environment = cuda_runtime_environment(environment)
    for key, value in overrides.items():
        section, option = key.rsplit(".", 1)
        if not settings.has_option(section, option):
            raise ValueError(f"Unknown upstream configuration key: {key}")
        settings[section][option] = str(value)
    if manifest.config.python_environment and manifest.config.python_environment.device_resident:
        if settings.getint("vec", "num_buffers") != 1:
            raise ValueError("Device-resident Python environments require one GPU buffer")
        if "base.cudagraphs" in overrides and settings.getint("base", "cudagraphs") != -1:
            raise ValueError("Device-resident Python callbacks do not support CUDA graph capture")
        settings["base"]["cudagraphs"] = "-1"
    if manifest.config.python_environment:
        agents = settings.getint("vec", "total_agents")
        buffers = settings.getint("vec", "num_buffers")
        per_environment = manifest.config.python_environment.spec.agents
        if agents <= 0 or buffers <= 0 or agents % (buffers * per_environment):
            raise ValueError("Each native buffer must contain a positive whole number of Python environments")
        if manifest.config.python_environment.device_resident and (buffers != 1 or agents != per_environment):
            raise ValueError("Device-resident Python environments require one GPU buffer with all agents")
    validate_minibatch_rotation(settings)
    batch_steps = settings.getint("vec", "total_agents") * settings.getint("train", "horizon")
    world_size = settings.getint("train", "gpus")
    environment_count = (
        settings.getint("vec", "total_agents") * world_size // manifest.config.python_environment.spec.agents
        if manifest.config.python_environment and not manifest.config.python_environment.device_resident
        else 0
    )
    if batch_steps <= 0 or world_size <= 0 or config.total_timesteps < batch_steps * world_size:
        raise ValueError("Training budget must include at least one positive rollout batch per GPU")
    expected_step = config.total_timesteps // (batch_steps * world_size) * batch_steps
    if len(output.name.encode()) >= 64:
        raise ValueError("Puffer run directory name must fit its 63-byte run ID")
    initial_parameters = b""
    initial_learner = b""
    initialization = None
    environment["METTA_INITIAL_POLICY"] = ""
    environment["METTA_INITIAL_EMA"] = ""
    environment["METTA_INITIAL_HORDE"] = ""
    environment["METTA_INITIAL_RND"] = ""
    environment["METTA_INITIAL_LEARNER"] = ""
    environment["METTA_INITIAL_ENVIRONMENT"] = ""
    environment["METTA_PYTHON_EXECUTABLE"] = sys.executable
    environment["METTA_RUN_RECORD"] = str(output / "training.json")
    if config.initialize:
        from integrations.native_spatial_asset import load_asset, training_contract
        from integrations.learner_checkpoint import LearnerCheckpoint as NativeLearnerCheckpoint
        from integrations.classic_contract import validate_training_contract
        import importlib

        reference = config.initialize
        asset = load_asset(reference.asset, manifest_sha256=reference.manifest_sha256)
        fabric = manifest.config.fabric
        if fabric is None or manifest.config.python_environment is None:
            raise ValueError("Native spatial initialization requires the current Fabric and Python environment")
        validate_training_contract(manifest.config.model_dump(mode="json"), config.model_dump(mode="json"))
        factory_module = importlib.import_module(fabric.factory.split(":", 1)[0])
        factory_sha256 = hashlib.sha256(Path(factory_module.__file__).read_bytes()).hexdigest()
        # The actual native actor admits its ABI before checkpoint loading.
        if (asset.metadata["factory_source_sha256"] != factory_sha256
                or asset.metadata["model_sha256"] != manifest.model_sha256
                or fabric_fingerprint(fabric) != manifest.model_sha256
                or FabricConfig.model_validate(asset.metadata["fabric"]) != fabric):
            raise ValueError("Native asset factory, model or Fabric configuration differs")
        initial_parameters = asset.policy
        initialization = InitializationRecord(
            asset=reference.asset.resolve(), manifest_sha256=reference.manifest_sha256,
            checkpoint_sha256=asset.metadata["policy_sha256"],
            training_seeds=asset.metadata["training_seeds"], restore_learner=reference.restore_learner,
        )
        environment["METTA_INITIAL_POLICY"] = str(output / "initial-policy.bin")
        if reference.restore_learner:
            if world_size != 1 or settings.getint("base", "async", fallback=0):
                raise ValueError("Learner restore requires synchronous single-GPU training")
            if environment_count or settings.getint("selfplay", "enabled", fallback=0):
                raise ValueError("Current native spatial assets restore device-resident learner state only")
            if asset.learner is None:
                raise ValueError("Native asset has no authentic learner state to restore")
            if asset.metadata["learner_configuration"] != {"seed": config.seed, "overrides": config.overrides}:
                raise ValueError("Learner restore requires identical seed and training overrides")
            objective = training_contract(manifest.config.python_environment.options, config.overrides)
            if asset.metadata["training_contract"] != objective:
                raise ValueError("Learner restore requires the identical current game and reward objective")
            learner = NativeLearnerCheckpoint.from_bytes(asset.learner, len(initial_parameters) // 4)
            if learner.epoch * batch_steps != learner.agent_steps:
                raise ValueError("Learner counters differ from the actual target rollout geometry")
            if learner.agent_steps >= expected_step:
                raise ValueError("Restore budget must leave at least one complete rollout")
            initial_learner = asset.learner
            environment["METTA_INITIAL_LEARNER"] = str(output / "initial-policy.bin.learner")
    return PreparedRun(
        build=build,
        output=output,
        name=name if name is not None else output.name,
        manifest=manifest,
        config=config,
        overrides=overrides,
        environment=environment,
        expected_step=expected_step,
        world_size=world_size,
        batch_steps=batch_steps,
        environment_count=environment_count,
        initial_parameters=initial_parameters,
        initialization=initialization,
        initial_learner=initial_learner,
    )


def create_run_monitor(run: PreparedRun) -> RunMonitor:
    now = datetime.now(UTC)
    return RunMonitor(
        run.output,
        RunSnapshot(
            name=run.name,
            model=f"puffer5/{run.manifest.config.environment}",
            method="Puffer 5 PPO",
            started_at=now,
            updated_at=now,
            progress_unit="agent_steps",
            max_steps=run.config.total_timesteps,
            summary_metric="env/perf",
        ),
    )


def initialize_run(run: PreparedRun, monitor: RunMonitor) -> None:
    monitor.set_phase("training")
    monitor.set_training_run_id(monitor.snapshot.run_id)
    RunIdentity(run_id=monitor.snapshot.run_id, name=run.name, backend="puffer5").write(run.output)
    (run.output / "training.json").write_text(
        TrainingRecord(build=run.manifest, config=run.config).model_dump_json(indent=2) + "\n"
    )
    if run.manifest.config.fabric:
        from integrations import native_startup_admission
        run.environment["METTA_NATIVE_ADMISSION_RUN_SHA256"] = native_startup_admission.digest(run.output / "training.json")
        run.environment["METTA_NATIVE_ADMISSION_SOURCE_SHA256"] = native_startup_admission.digest(native_startup_admission.__file__)
    if run.initialization:
        (run.output / "initial-policy.bin").write_bytes(run.initial_parameters)
        (run.output / "initialization.json").write_text(run.initialization.model_dump_json(indent=2) + "\n")
    if run.initial_learner:
        (run.output / "initial-policy.bin.learner").write_bytes(run.initial_learner)
    if run.manifest.config.python_environment:
        run.environment.update(
            environment_runtime(
                run.manifest.config.python_environment,
                run.manifest.environment_sha256,
                run.output,
                run.config.seed,
                "train",
            )
        )


def finish_run(run: PreparedRun, monitor: RunMonitor) -> TrainingResult:
    output, manifest = run.output, run.manifest
    if manifest.config.fabric:
        from integrations.native_startup_admission import digest, verify_receipt
        if digest(output / "training.json") != run.environment["METTA_NATIVE_ADMISSION_RUN_SHA256"]:
            raise ValueError("Native admission training record changed")
        verify_receipt(output / "training.json")
    checkpoints = sorted((output / "checkpoints").rglob("*.bin"))
    final_checkpoint = (
        output / "checkpoints" / manifest.config.environment / output.name / f"{run.expected_step:016d}.bin"
    )
    if not final_checkpoint.is_file() or final_checkpoint.stat().st_size == 0:
        raise RuntimeError("Puffer exited without its final nonempty policy checkpoint")
    for checkpoint in checkpoints:
        policy = checkpoint.read_bytes()
        if not policy or len(policy) % 4:
            raise ValueError("Puffer checkpoint must contain nonempty float32 policy weights")
        state_path = Path(str(checkpoint) + ".learner")
        learner = LearnerCheckpoint.read(state_path, len(policy) // 4)
        if learner.agent_steps != int(checkpoint.stem) or learner.epoch * run.batch_steps != learner.agent_steps:
            raise ValueError("Learner snapshot counters differ from its checkpoint or rollout dimensions")
        identity = LearnerCheckpointIdentity.model_validate_json(Path(str(state_path) + ".json").read_text())
        if (
            identity.policy_sha256 != hashlib.sha256(policy).hexdigest()
            or identity.state_sha256 != hashlib.sha256(state_path.read_bytes()).hexdigest()
            or identity.run_sha256 != hashlib.sha256((output / "training.json").read_bytes()).hexdigest()
            or len(identity.environment_sha256) != run.environment_count
        ):
            raise ValueError("Published checkpoint identity differs from its run, policy, or learner state")
        for index, digest in enumerate(identity.environment_sha256):
            snapshot = Path(f"{checkpoint}.environment.{index}.json").read_bytes()
            if (
                hashlib.sha256(snapshot).hexdigest() != digest
                or NativeEnvironmentCheckpoint.model_validate_json(snapshot).index != index
            ):
                raise ValueError("Published environment snapshot differs from its checkpoint identity")
    log_path = output / "logs" / manifest.config.environment / f"{output.name}.ini"
    metrics = ConfigParser(interpolation=None)
    with log_path.open() as result:
        metrics.read_file(result)
    if not metrics.has_option("metrics", "agent_steps"):
        raise RuntimeError("Puffer exited without timestep metrics")
    if float(metrics["metrics"]["agent_steps"].split(",")[-1]) != run.expected_step * run.world_size:
        raise RuntimeError("Puffer timestep metrics differ from the requested final learner step")
    monitor.record(
        run.expected_step * run.world_size,
        {key: float(value.split(",")[-1]) for key, value in metrics["metrics"].items()},
    )
    result = TrainingResult(
        checkpoints=[path.relative_to(output) for path in checkpoints],
        final_checkpoint=final_checkpoint.relative_to(output),
        revision=manifest.revision,
        trained_timesteps=run.expected_step * run.world_size,
        metrics=log_path.relative_to(output),
    )
    identity = RunIdentity.model_validate_json((output / "run.json").read_text())
    TrainingArtifact(
        run_id=identity.run_id,
        uri=final_checkpoint.as_uri(),
        step=result.trained_timesteps,
        progress_unit="agent_steps",
        resumable=False,
    ).write(output)
    (output / "completed.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def train_puffer(
    build: Path, output: Path, config: RunConfig, *, name: str | None = None, timeout_seconds: float | None = None
) -> TrainingResult:
    """Run upstream training with bounded steps and an isolated artifact directory."""
    run = prepare_run(build, output, config, name=name)
    with create_run_monitor(run) as monitor:
        initialize_run(run, monitor)
        command = [
            str(run.build / "puffer"),
            "train",
            f"--train.total_timesteps={config.total_timesteps}",
            f"--base.seed={config.seed}",
            f"--base.checkpoint_dir={run.output / 'checkpoints'}",
            f"--base.log_dir={run.output / 'logs'}",
            f"--base.run_id={run.output.name}",
            *[f"--{key}={value}" for key, value in sorted(run.overrides.items())],
        ]
        with (run.output / "console.log").open("x") as log:
            subprocess.run(
                command,
                cwd=run.build / "source",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=run.environment,
                timeout=timeout_seconds,
            )
        return finish_run(run, monitor)

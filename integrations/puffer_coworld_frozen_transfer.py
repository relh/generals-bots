"""Build and execute a pinned upstream PufferLib engine."""

import hashlib
import math
import os
import struct
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
    EMAState,
    FabricConfig,
    HordeCheckpointState,
    ObjectiveSettings,
    RNDCheckpointState,
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
    environment_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")

    @model_validator(mode="after")
    def validate_model_identity(self) -> "BuildManifest":
        if bool(self.config.python_environment) != bool(self.environment_sha256):
            raise ValueError("Python environments require an implementation fingerprint")
        if self.config.fabric:
            if not self.model_sha256 or self.model_state_words == 0:
                raise ValueError("Fabric builds require a model fingerprint and recurrent-state allocation")
        elif self.model_sha256 or self.model_state_words:
            raise ValueError("Native models cannot declare Fabric metadata")
        return self


def verified_v11_policy_transfer(source: BuildManifest, target: BuildManifest, checkpoint_sha256: str) -> bool:
    """Permit only the archived v11 sparse-teacher policy into its dense-teacher GPU graph."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    if source_env is None or target_env is None or source.config.fabric is None or target.config.fabric is None:
        return False
    if (
        source.model_sha256 != "c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9"
        or target.model_sha256
        not in {
            "5ba4d2d8d83cf6f7ae3276269d9f39519a502857d6a730577f7a80dbd81e247f",
            "a8fdf4cef1cbdb157206b237959234d71c37fb0a36ff4ee5289464f66d7333f4",
        }
        or checkpoint_sha256 != "939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499"
        or source.model_state_words != 12360
        or target.model_state_words != 12360
        or source.revision != target.revision
        or source_env.factory != target_env.factory
        or source_env.spec.observation_size != target_env.spec.observation_size
        or source_env.spec.action_sizes != target_env.spec.action_sizes
        or source_env.spec.replay_metadata_size != 2
        or source_env.spec.teacher
        or target_env.spec.replay_metadata_size != 0
        or not target_env.spec.teacher
        or source_env.device_resident
        or not target_env.device_resident
    ):
        return False
    if source_env.model_dump(exclude={"options", "spec", "device_resident"}) != target_env.model_dump(
        exclude={"options", "spec", "device_resident"}
    ):
        return False
    if source_env.spec.model_dump(exclude={"agents", "replay_metadata_size", "teacher"}) != target_env.spec.model_dump(
        exclude={"agents", "replay_metadata_size", "teacher"}
    ):
        return False
    source_options = dict(source_env.options)
    target_options = dict(target_env.options)
    source_changes = {"opponent": "mixed", "parallel_games": 1024, "sparse_teacher": True, "supervise_teacher": False}
    target_changes = {
        "opponent": "strong_mixed",
        "parallel_games": 4096,
        "sparse_teacher": False,
        "supervise_teacher": True,
        "imitation_weight": 0.0,
    }
    if any(source_options.pop(name, None) != value for name, value in source_changes.items()):
        return False
    if any(target_options.pop(name, None) != value for name, value in target_changes.items()):
        return False
    if source_options != target_options:
        return False
    source_build = source.config.model_dump(exclude={"python_environment"})
    target_build = target.config.model_dump(exclude={"python_environment"})
    source_build["environment_backend"] = target_build["environment_backend"]
    for config in (source_build, target_build):
        fabric = config["fabric"]
        for field in ("teacher", "losses", "replay_metadata_size"):
            fabric.pop(field)
    return source_build == target_build


def verified_classic_selfplay_policy_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Allow only the frozen-logit-parity-audited Classic policy into two-seat training."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    source_fabric, target_fabric = source.config.fabric, target.config.fabric
    if source_env is None or target_env is None or source_fabric is None or target_fabric is None:
        return False
    if (
        source.model_sha256 != "2f075e32c9979c3a550608f015fcef401f1ac1b8b50d02bc67b7a4636c163e8d"
        or target.model_sha256 != "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
        or checkpoint_sha256 != "c6cc6db63b3b5855d1ab7f38b99baf8ed25cfcb51f2c34819a43a1d58b51fe3a"
        or source.model_state_words != target.model_state_words
        or source.model_state_words != 22956
        or source.revision != target.revision
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsPufferEnvironment"
        or target_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or source_env.spec != target_env.spec
        or source_env.spec.agents != 8192
        or source_env.spec.observation_size != 14 * 21 * 21
        or source_env.spec.action_sizes != [1765, 2]
        or source_fabric.model_dump(exclude={"teacher"}) != target_fabric.model_dump(exclude={"teacher"})
        or source.config.model_dump(exclude={"python_environment", "fabric"})
        != target.config.model_dump(exclude={"python_environment", "fabric"})
        or source_env.model_dump(exclude={"factory", "options"})
        != target_env.model_dump(exclude={"factory", "options"})
    ):
        return False
    source_options = dict(source_env.options)
    target_options = dict(target_env.options)
    if source_options.pop("parallel_games", None) != 8192:
        return False
    if target_options.pop("parallel_games", None) != 4096:
        return False
    return source_options == target_options


def verified_classic_iter1_calibrated_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Permit one parity-audited self-play checkpoint into the calibrated PPO build."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    source_fabric, target_fabric = source.config.fabric, target.config.fabric
    if source_env is None or target_env is None or source_fabric is None or target_fabric is None:
        return False
    if (
        source.model_sha256 != "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
        or target.model_sha256 != "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
        or checkpoint_sha256 != "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf"
        or source.model_state_words != target.model_state_words
        or source.model_state_words != 22956
        or source.revision != target.revision
        or source_env.factory != target_env.factory
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or source_env.spec != target_env.spec
        or source_env.spec.agents != 8192
        or source_env.spec.observation_size != 14 * 21 * 21
        or source_env.spec.action_sizes != [1765, 2]
        or source_fabric.model_dump(exclude={"teacher"}) != target_fabric.model_dump(exclude={"teacher"})
        or source.config.model_dump(exclude={"python_environment", "fabric"})
        != target.config.model_dump(exclude={"python_environment", "fabric"})
        or source_env.model_dump(exclude={"options"}) != target_env.model_dump(exclude={"options"})
    ):
        return False
    source_options = dict(source_env.options)
    target_options = dict(target_env.options)
    if "move_hint_scale" in source_options or "split_hint_scale" in source_options:
        return False
    if target_options.pop("move_hint_scale", None) != 0.25:
        return False
    if target_options.pop("split_hint_scale", None) != 0.25:
        return False
    if source_options != target_options:
        return False
    source_phases = source_fabric.model_dump()["teacher"]["phases"]
    target_phases = target_fabric.model_dump()["teacher"]["phases"]
    if [phase["action_mix"] for phase in source_phases] != [1.0, 0.0]:
        return False
    return len(target_phases) == 1 and all(
        target_phases[0][key] == value
        for key, value in {
            "agent_steps": 0, "ppo_coefficient": 1.0, "coefficient": 0.0, "action_mix": 0.0
        }.items()
    )


def verified_classic_frozen_opponent_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Move the pinned generation-1 learner into one-seat frozen-opponent training."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    if source_env is None or target_env is None:
        return False
    if (
        checkpoint_sha256 != "d27cb8552a78b083201575a604a1c3f6b2c43336f4e32e45221cf3d11b77a032"
        or source.model_sha256 != target.model_sha256
        or source.model_sha256 != "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
        or source.model_state_words != target.model_state_words
        or source.revision != target.revision
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or target_env.factory != "integrations.metta_puffer:BatchedGeneralsFrozenOpponentPufferEnvironment"
        or source_env.spec.agents != 8192
        or target_env.spec.agents != 4096
        or source_env.spec.model_copy(update={"agents": 4096}) != target_env.spec
        or source_env.model_dump(exclude={"factory", "spec", "options"})
        != target_env.model_dump(exclude={"factory", "spec", "options"})
        or source.config.model_dump(exclude={"python_environment"})
        != target.config.model_dump(exclude={"python_environment"})
    ):
        return False
    options = dict(target_env.options)
    if options.pop("frozen_sha256", None) != "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf":
        return False
    if options.pop("frozen_build", None) != "/recovery/classic-selfplay-teacher-h128-build-27857/build.json":
        return False
    if options.pop("frozen_checkpoint", None) != (
        "/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin"
    ):
        return False
    if options.pop("scripted_hint_fraction", 0.0) not in (0.0, 0.5):
        return False
    return options == source_env.options


def verified_classic_flat_gen0_opponent_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Initialize the flat learner from its early checkpoint against frozen generation 0."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    if source_env is None or target_env is None:
        return False
    if (
        checkpoint_sha256 != "5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71"
        or source.model_sha256 != target.model_sha256
        or source.model_sha256 != "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
        or source.model_state_words != target.model_state_words
        or source.revision != target.revision
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or target_env.factory != "integrations.metta_puffer:BatchedGeneralsFrozenOpponentPufferEnvironment"
        or source_env.spec.agents != 8192
        or target_env.spec.agents != 4096
        or source_env.spec.model_copy(update={"agents": 4096}) != target_env.spec
        or source_env.model_dump(exclude={"factory", "spec", "options"})
        != target_env.model_dump(exclude={"factory", "spec", "options"})
        or source.config.model_dump(exclude={"python_environment"})
        != target.config.model_dump(exclude={"python_environment"})
    ):
        return False
    options = dict(target_env.options)
    required = {
        "frozen_sha256": "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf",
        "frozen_build": "/recovery/classic-selfplay-teacher-h128-build-27857/build.json",
        "frozen_checkpoint": (
            "/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/"
            "metta_generals/run/0000000033554432.bin"
        ),
        "frozen_codec": "hinted_gen0",
        "frozen_legacy_fabric": (
            "/recovery/classic-flat-gen0-frozen-audit-29311/staged/legacy/generals_fabric.py"
        ),
    }
    if any(options.pop(key, None) != value for key, value in required.items()):
        return False
    return options == source_env.options


def verified_classic_flat_scripted_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Move the pinned early flat actor from two-seat play to scripted one-seat play."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    if source_env is None or target_env is None:
        return False
    return (
        checkpoint_sha256 == "5303af89afaa579657c0254eb29754c9cc7638ef86134b9a0eaadccbc451fd71"
        and source.model_sha256 == target.model_sha256
        and source.model_sha256 == "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
        and source.model_state_words == target.model_state_words
        and source.revision == target.revision
        and source_env.factory == "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        and target_env.factory == "integrations.metta_puffer:BatchedGeneralsPufferEnvironment"
        and source_env.spec.agents == 8192
        and target_env.spec.agents == 4096
        and source_env.spec.model_copy(update={"agents": 4096}) == target_env.spec
        and source_env.model_dump(exclude={"factory", "spec"})
        == target_env.model_dump(exclude={"factory", "spec"})
        and source.config.model_dump(exclude={"python_environment"})
        == target.config.model_dump(exclude={"python_environment"})
    )


def verified_classic_gen0_frozen_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Start one-seat PPO from the parity-audited unscaled generation-0 actor."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    source_fabric, target_fabric = source.config.fabric, target.config.fabric
    if source_env is None or target_env is None or source_fabric is None or target_fabric is None:
        return False
    if (
        checkpoint_sha256 != "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf"
        or source.model_sha256 != "a5a48d16d5c44f057f8c8323b6c531ccce6f06de26a0a47cefe4d68f527de2dd"
        or target.model_sha256 != "4f718ad75a43d99e6c33de5a23bdc553443b243f23ad5268f66d9f276bf10a67"
        or source.model_state_words != target.model_state_words
        or source.model_state_words != 22956
        or source.revision != target.revision
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or target_env.factory != "integrations.metta_puffer:BatchedGeneralsFrozenOpponentPufferEnvironment"
        or source_env.spec.agents != 8192
        or target_env.spec.agents != 4096
        or source_env.spec.model_copy(update={"agents": 4096}) != target_env.spec
        or source_env.model_dump(exclude={"factory", "spec", "options"})
        != target_env.model_dump(exclude={"factory", "spec", "options"})
        or source_fabric.model_dump(exclude={"teacher"}) != target_fabric.model_dump(exclude={"teacher"})
        or source.config.model_dump(exclude={"python_environment", "fabric"})
        != target.config.model_dump(exclude={"python_environment", "fabric"})
    ):
        return False
    options = dict(target_env.options)
    if options.pop("frozen_sha256", None) != checkpoint_sha256:
        return False
    if options.pop("frozen_build", None) != "/recovery/classic-selfplay-teacher-h128-build-27857/build.json":
        return False
    if options.pop("frozen_checkpoint", None) != (
        "/recovery/classic-selfplay-init134-h128-27957/run/checkpoints/metta_generals/run/0000000033554432.bin"
    ):
        return False
    if options.pop("scripted_hint_fraction", 0.0) != 0.0:
        return False
    # The reward-clamp pilot keeps the actor and game codec unchanged. Permit
    # only its pinned reward rescaling, then compare every remaining option.
    if options.get("reward_scale") == 0.02 and source_env.options.get("reward_scale") == 0.5:
        options["reward_scale"] = 0.5
    if options != source_env.options:
        return False
    source_phases = source_fabric.model_dump()["teacher"]["phases"]
    target_phases = target_fabric.model_dump()["teacher"]["phases"]
    if [phase["action_mix"] for phase in source_phases] != [1.0, 0.0]:
        return False
    return len(target_phases) == 1 and all(
        target_phases[0][key] == value
        for key, value in {
            "agent_steps": 0, "ppo_coefficient": 1.0, "coefficient": 0.0, "action_mix": 0.0
        }.items()
    )


def verified_classic_nohint_dagger_transfer(
    source: BuildManifest, target: BuildManifest, checkpoint_sha256: str
) -> bool:
    """Carry the exact hint-free actor into a changed teacher action schedule."""
    source_env, target_env = source.config.python_environment, target.config.python_environment
    source_fabric, target_fabric = source.config.fabric, target.config.fabric
    if source_env is None or target_env is None or source_fabric is None or target_fabric is None:
        return False
    if (
        checkpoint_sha256 != "f075b076cb428b69049e8ce3975284f5b69f837e27c94ec7c59541d26aaf0609"
        or source.model_sha256 != "327d1aee8ae7245c60377667f65cbc6d52d1f30f87b890efc52f042a8c77ecb9"
        or target.model_sha256 != "f64e6030af8ef5f01a44d067f4ac31cd9573c02296a9b7872dce5b33b2ffe627"
        or source.model_state_words != target.model_state_words
        or source.model_state_words != 17664
        or source.revision != target.revision
        or source_env.factory != "integrations.metta_puffer:BatchedGeneralsSelfPlayPufferEnvironment"
        or source_env != target_env
        or source_env.spec.agents != 8192
        or source_env.spec.observation_size != 8 * 21 * 21
        or source_env.spec.action_sizes != [1765, 2]
        or source_fabric.model_dump(exclude={"teacher"}) != target_fabric.model_dump(exclude={"teacher"})
        or source.config.model_dump(exclude={"fabric"}) != target.config.model_dump(exclude={"fabric"})
    ):
        return False
    source_phases = source_fabric.model_dump()["teacher"]["phases"]
    target_phases = target_fabric.model_dump()["teacher"]["phases"]
    return (
        len(source_phases) == len(target_phases) == 2
        and source_phases[0]["action_mix"] == 1.0
        and source_phases[1]["agent_steps"] == 33_554_432
        and target_phases[0]["action_mix"] == 0.5
        and target_phases[1]["agent_steps"] == 100_663_296
        and all(
            {key: value for key, value in source_phase.items() if key not in ("agent_steps", "action_mix")}
            == {key: value for key, value in target_phase.items() if key not in ("agent_steps", "action_mix")}
            for source_phase, target_phase in zip(source_phases, target_phases, strict=True)
        )
    )


class CheckpointInitialization(Configuration):
    """Verified policy initialization, optionally restoring optimizer and learner clocks."""

    run: Path
    checkpoint: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allow_environment_transfer: bool = False
    allow_policy_only_transfer: bool = False
    restore_ema: bool = False
    restore_horde: bool = False
    restore_rnd: bool = False
    restore_learner: bool = False


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
    source: TrainingRecord
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_seeds: list[int] = Field(min_length=1)


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
    initial_teacher: EMAState | None = None
    initial_horde: HordeCheckpointState | None = None
    initial_rnd: RNDCheckpointState | None = None
    initial_learner: bytes = Field(default=b"", repr=False)
    initial_environments: list[bytes] = Field(default_factory=list, repr=False)
    initialization: InitializationRecord | None = None


def build_puffer(output: Path, config: BuildConfig) -> BuildManifest:
    """Build in an exclusive directory; never modify another checkout or build."""
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
    model_digest = fabric_fingerprint(config.fabric) if config.fabric else ""
    with (output / "build.log").open("x") as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    manifest = BuildManifest(
        config=config,
        binary_sha256=hashlib.sha256((output / "puffer").read_bytes()).hexdigest(),
        model_sha256=model_digest,
        model_state_words=state_words,
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
    initial_teacher = None
    initial_horde = None
    initial_rnd = None
    initial_learner = b""
    initial_environments = []
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
        reference = config.initialize
        source_run = reference.run.resolve()
        checkpoint = reference.checkpoint.resolve()
        source = TrainingRecord.model_validate_json((source_run / "training.json").read_text())
        if not checkpoint.is_relative_to(source_run / "checkpoints"):
            raise ValueError("Initialization checkpoint must belong to its source run")
        if not reference.restore_learner:
            completed = TrainingResult.model_validate_json((source_run / "completed.json").read_text())
            if checkpoint.relative_to(source_run) not in completed.checkpoints:
                raise ValueError("Initialization checkpoint is absent from the completed run")
        classic_selfplay_transfer = reference.allow_policy_only_transfer and verified_classic_selfplay_policy_transfer(
            source.build, manifest, reference.sha256
        )
        gen0_frozen_transfer = verified_classic_gen0_frozen_transfer(
            source.build, manifest, reference.sha256
        )
        frozen_transfer = (
            verified_classic_frozen_opponent_transfer(source.build, manifest, reference.sha256)
            or verified_classic_flat_gen0_opponent_transfer(source.build, manifest, reference.sha256)
            or verified_classic_flat_scripted_transfer(source.build, manifest, reference.sha256)
            or gen0_frozen_transfer
        )
        policy_only_transfer = reference.allow_policy_only_transfer and (
            classic_selfplay_transfer
            or verified_classic_iter1_calibrated_transfer(source.build, manifest, reference.sha256)
            or verified_v11_policy_transfer(source.build, manifest, reference.sha256)
            or gen0_frozen_transfer
            or verified_classic_nohint_dagger_transfer(source.build, manifest, reference.sha256)
        )
        if reference.allow_policy_only_transfer and (
            not reference.allow_environment_transfer
            or reference.restore_learner
            or reference.restore_ema
            or reference.restore_horde
            or reference.restore_rnd
            or not policy_only_transfer
        ):
            raise ValueError("Policy-only transfer requires a verified checkpoint and GPU target")
        if reference.allow_environment_transfer:
            source_env = source.build.config.python_environment
            target_env = manifest.config.python_environment
            compatible_environment = (
                source_env is not None
                and target_env is not None
                and (source_env.factory == target_env.factory or classic_selfplay_transfer or frozen_transfer)
                and (
                    source_env.spec.model_copy(update={"agents": target_env.spec.agents}) == target_env.spec
                    or policy_only_transfer
                )
                and (
                    source.build.config.model_dump(exclude={"python_environment"})
                    == manifest.config.model_dump(exclude={"python_environment"})
                    or policy_only_transfer
                )
                and source.build.revision == manifest.revision
            )
        else:
            compatible_environment = (
                source.build.config == manifest.config
                and source.build.environment_sha256 == manifest.environment_sha256
            )
        if (
            not compatible_environment
            or (source.build.model_sha256 != manifest.model_sha256 and not policy_only_transfer)
            or source.build.model_state_words != manifest.model_state_words
        ):
            raise ValueError("Initialization requires matching model and compatible environment configuration")
        source_policy = {key: value for key, value in source.config.overrides.items() if key.startswith("policy.")}
        target_policy = {key: value for key, value in config.overrides.items() if key.startswith("policy.")}
        if source_policy != target_policy:
            raise ValueError("Initialization requires identical policy architecture overrides")
        initial_parameters = checkpoint.read_bytes()
        if hashlib.sha256(initial_parameters).hexdigest() != reference.sha256:
            raise ValueError("Initialization checkpoint digest differs from its declared identity")
        if not initial_parameters or len(initial_parameters) % 4:
            raise ValueError("Initialization requires a nonempty float32 checkpoint")
        if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", initial_parameters)):
            raise ValueError("Initialization checkpoint contains nonfinite parameters")
        seeds = training_lineage_seeds(source_run, source)
        initialization = InitializationRecord(
            source=source, checkpoint_sha256=reference.sha256, training_seeds=sorted(seeds)
        )
        environment["METTA_INITIAL_POLICY"] = str(output / "initial-policy.bin")
        fabric = manifest.config.fabric
        if reference.restore_learner:
            if world_size != 1 or settings.getint("base", "async", fallback=0):
                raise ValueError("Learner resume currently requires synchronous single-GPU training")
            if settings.getint("selfplay", "enabled", fallback=0):
                raise ValueError("Learner resume requires opponent history recovery for self-play")
            if source.config.overrides != config.overrides or source.config.seed != config.seed:
                raise ValueError("Learner resume requires the identical native build, seed, and training overrides")
            state_path = Path(str(checkpoint) + ".learner")
            identity = LearnerCheckpointIdentity.model_validate_json(Path(str(state_path) + ".json").read_text())
            initial_learner = state_path.read_bytes()
            if (
                identity.policy_sha256 != reference.sha256
                or identity.state_sha256 != hashlib.sha256(initial_learner).hexdigest()
                or identity.run_sha256 != hashlib.sha256((source_run / "training.json").read_bytes()).hexdigest()
                or len(identity.environment_sha256) != environment_count
            ):
                raise ValueError("Learner snapshot differs from its recorded policy or optimizer identity")
            for index, digest in enumerate(identity.environment_sha256):
                snapshot = Path(f"{checkpoint}.environment.{index}.json").read_bytes()
                state = NativeEnvironmentCheckpoint.model_validate_json(snapshot)
                if hashlib.sha256(snapshot).hexdigest() != digest or state.index != index:
                    raise ValueError("Environment snapshot differs from its recorded identity")
                if state.snapshot is None:
                    raise ValueError("Learner resume requires restorable environment snapshots")
                initial_environments.append(snapshot)
            if initial_environments:
                environment["METTA_INITIAL_ENVIRONMENT"] = str(output / "initial-policy.bin")
            learner = LearnerCheckpoint.read(state_path, len(initial_parameters) // 4)
            if learner.epoch * batch_steps != learner.agent_steps or learner.agent_steps != int(checkpoint.stem):
                raise ValueError("Learner snapshot counters differ from its checkpoint or rollout dimensions")
            if learner.agent_steps >= expected_step:
                raise ValueError("Resume budget must leave at least one complete rollout after the checkpoint")
            environment["METTA_INITIAL_LEARNER"] = str(output / "initial-policy.bin.learner")
        if reference.restore_ema or (
            reference.restore_learner and fabric and (fabric.self_distillation or fabric.ema_prior)
        ):
            if not manifest.config.fabric or not (
                manifest.config.fabric.self_distillation or manifest.config.fabric.ema_prior
            ):
                raise ValueError("Restoring an EMA prior requires an EMA model configuration")
            initial_teacher = EMAState.model_validate_json(Path(str(checkpoint) + ".ema.json").read_text())
            if (
                initial_teacher.student_sha256 != reference.sha256
                or initial_teacher.configuration_sha256
                != hashlib.sha256(manifest.config.fabric.model_dump_json().encode()).hexdigest()
                or 4 * len(initial_teacher.parameters) != len(initial_parameters)
            ):
                raise ValueError("EMA snapshot differs from the configured model or student checkpoint")
            environment["METTA_INITIAL_EMA"] = str(output / "initial-policy.bin")
        if reference.restore_horde or (reference.restore_learner and fabric and fabric.horde):
            if not manifest.config.fabric or not manifest.config.fabric.horde:
                raise ValueError("Restoring Horde requires its collection configuration")
            initial_horde = HordeCheckpointState.model_validate_json(Path(str(checkpoint) + ".horde.json").read_text())
            if (
                initial_horde.student_sha256 != reference.sha256
                or initial_horde.configuration_sha256
                != hashlib.sha256(manifest.config.fabric.model_dump_json().encode()).hexdigest()
            ):
                raise ValueError("Horde snapshot differs from the configured model or student checkpoint")
            environment["METTA_INITIAL_HORDE"] = str(output / "initial-policy.bin")
        if reference.restore_rnd or (reference.restore_learner and fabric and fabric.rnd):
            if not manifest.config.fabric or not manifest.config.fabric.rnd:
                raise ValueError("Restoring RND requires its collection configuration")
            initial_rnd = RNDCheckpointState.model_validate_json(Path(str(checkpoint) + ".rnd.json").read_text())
            if (
                initial_rnd.student_sha256 != reference.sha256
                or initial_rnd.configuration_sha256
                != hashlib.sha256(manifest.config.fabric.model_dump_json().encode()).hexdigest()
            ):
                raise ValueError("RND snapshot differs from the configured model or student checkpoint")
            environment["METTA_INITIAL_RND"] = str(output / "initial-policy.bin")
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
        initial_teacher=initial_teacher,
        initial_horde=initial_horde,
        initial_rnd=initial_rnd,
        initial_learner=initial_learner,
        initial_environments=initial_environments,
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
    if run.initialization:
        (run.output / "initial-policy.bin").write_bytes(run.initial_parameters)
        (run.output / "initialization.json").write_text(run.initialization.model_dump_json(indent=2) + "\n")
    if run.initial_teacher:
        (run.output / "initial-policy.bin.ema.json").write_text(run.initial_teacher.model_dump_json() + "\n")
    if run.initial_learner:
        (run.output / "initial-policy.bin.learner").write_bytes(run.initial_learner)
    for index, snapshot in enumerate(run.initial_environments):
        (run.output / f"initial-policy.bin.environment.{index}.json").write_bytes(snapshot)
    if run.initial_rnd:
        (run.output / "initial-policy.bin.rnd.json").write_text(run.initial_rnd.model_dump_json() + "\n")
    if run.initial_horde:
        (run.output / "initial-policy.bin.horde.json").write_text(run.initial_horde.model_dump_json() + "\n")
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

"""Inspect the pinned METTAL01 Muon snapshot without importing a trainer.

This read-only format gate is used while preparing portable inputs on machines
that do not have the optional Metta training framework installed. Checkpoint
publication and optimizer restoration remain the training runtime's responsibility.
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

LEARNER_HEADER = struct.Struct('<8sQQQf')


@dataclass(frozen=True)
class LearnerCheckpoint:
    epoch: int
    agent_steps: int
    learning_rate: float
    parameter_count: int

    @classmethod
    def read(cls, path: Path, parameter_count: int) -> 'LearnerCheckpoint':
        return cls.from_bytes(path.read_bytes(), parameter_count)

    @classmethod
    def from_bytes(cls, data: bytes, parameter_count: int) -> 'LearnerCheckpoint':
        if len(data) < LEARNER_HEADER.size:
            raise ValueError('Learner checkpoint header is truncated')
        magic, epoch, steps, count, rate = LEARNER_HEADER.unpack_from(data)
        if magic != b'METTAL01':
            raise ValueError('Unknown learner checkpoint format')
        if count <= 0 or count != parameter_count or len(data) != LEARNER_HEADER.size + count * 4:
            raise ValueError('Learner checkpoint parameter count or payload size differs')
        momentum = struct.iter_unpack('<f', memoryview(data)[LEARNER_HEADER.size:])
        if not math.isfinite(rate) or rate < 0 or not all(math.isfinite(v) for (v,) in momentum):
            raise ValueError('Learner checkpoint contains invalid optimizer state')
        return cls(epoch, steps, rate, count)


class LearnerCheckpointIdentity(BaseModel):
    """Public checkpoint hashes required before publishing a portable asset."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_sha256: list[str]


class TrainingResult(BaseModel):
    """Completed native run record shared by publication and the optional trainer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoints: list[Path]
    final_checkpoint: Path
    revision: str
    trained_timesteps: int = Field(gt=0)
    metrics: Path

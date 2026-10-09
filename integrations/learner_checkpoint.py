"""Inspect the pinned METTAL01 Muon snapshot without importing a trainer.

This read-only format gate is used while preparing portable inputs on machines
that do not have the optional Metta training framework installed. Checkpoint
publication and optimizer restoration remain the training runtime's responsibility.
"""

import math
import struct
from dataclasses import dataclass
from pathlib import Path

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

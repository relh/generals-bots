"""Extend the local stencil while retaining every learned parameter and momentum."""

import hashlib
import math
import struct

import numpy as np

LEARNER_HEADER = struct.Struct("<8sQQQf")
MAPPING_SHA256 = "dd2d1dd66681be35a7d64865ba4bd097ebb40fb5c41b576f3eb0808b5321cc9a"


def qualified_mapping():
    """Reconstruct the mapping proven against both pinned native F32 layouts.

    This avoids assembling two multi-million-edge graphs again during every
    initialization. The caller must first verify both complete model hashes.
    The native CPU audit independently reconstructs and compares every index.
    """
    from integrations.spatial_context_geometry import context_offsets
    from integrations.spatial_muon_context import OFFSET, load_gather

    old_offsets, new_offsets = context_offsets(1.01), context_offsets(2.01)
    old = OFFSET + load_gather(1.01).reshape(32, 32, 5)
    new = OFFSET + load_gather(2.01).reshape(32, 32, 13)
    mapping = np.full(578860, -1, np.int32)
    mapping[:OFFSET] = np.arange(OFFSET)
    for index, offset in enumerate(old_offsets):
        mapping[new[:, :, new_offsets.index(offset)]] = old[:, :, index]
    mapping[OFFSET + 13312:] = np.arange(OFFSET + 5120, 570668)
    # Exact native alignment padding, not learned parameters.
    mapping[[138, 139, 150, 151, 578273, 578274, 578275,
             578277, 578278, 578279, 578281, 578282, 578283]] = -1
    if hashlib.sha256(mapping.astype("<i4").tobytes()).hexdigest() != MAPPING_SHA256:
        raise ValueError("Context mapping differs from the independently audited native layouts")
    return mapping


def parameter_mapping(old, new, old_buffers, new_buffers):
    """Return target-to-source indices; -1 denotes zero initialization or padding."""
    if (old.channels, old.features, old.global_features) != (new.channels, new.features, new.global_features):
        raise ValueError("Context extension must preserve all other model dimensions")
    f = old.features
    if old.context_kernel.shape != (3, 3, f, f) or new.context_kernel.shape != (5, 5, f, f):
        raise ValueError("Context extension requires the cross-to-radius-two geometry")
    pairs = [(old.context_kernel, new.context_kernel[1:4, 1:4])]
    names = ("input_kernel", "action_kernel", "local_weight", "local_bias", "context_weight",
             "context_bias", "global_weight", "global_bias", "global_kernel", "readout_kernel",
             "output_weight", "output_bias")
    pairs.extend((getattr(old, name), getattr(new, name)) for name in names)
    if len(old.priors) != len(new.priors):
        raise ValueError("Context extension changes direct priors")
    for (old_source, old_indices), (new_source, new_indices) in zip(old.priors, new.priors, strict=True):
        if not np.array_equal(old_source, new_source):
            raise ValueError("Context extension changes prior inputs")
        pairs.append((old_indices, new_indices))
    mapping = np.full(new_buffers.parameter_words, -1, np.int32)
    for source, target in pairs:
        if source.shape != target.shape:
            raise ValueError("Context extension changes a retained tensor shape")
        for src, dst in zip(source.reshape(-1), target.reshape(-1), strict=True):
            if src < 0:
                continue  # New diagonal neighbors in the central 3x3 stay zero.
            if dst < 0 or mapping[dst] not in (-1, src):
                raise ValueError("Context extension changes parameter sharing")
            mapping[dst] = src

    def slots(buffers):
        return np.concatenate([np.arange(p.offset, p.offset + p.size) for p in buffers.parameters])

    old_slots, new_slots = slots(old_buffers), slots(new_buffers)
    if not np.array_equal(np.unique(mapping[mapping >= 0]), np.sort(old_slots)):
        raise ValueError("Context transfer does not preserve every original parameter")
    if len(mapping[mapping >= 0]) != len(old_slots):
        raise ValueError("Context transfer duplicates an original parameter")
    new_context = new.context_kernel[new.context_kernel >= 0]
    new_only = np.setdiff1d(new_slots, np.flatnonzero(mapping >= 0))
    if len(new_only) != 8 * f * f or not np.isin(new_only, new_context).all():
        raise ValueError("Unexpected parameters introduced by context extension")
    return mapping


def extend_flat(values, mapping):
    """Apply the same semantic mapping to policy weights or Muon momentum."""
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Expected finite float32 checkpoint parameters")
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < -1):
        raise ValueError("Invalid context parameter mapping")
    present = mapping >= 0
    if np.any(mapping[present] >= values.size):
        raise ValueError("Context mapping exceeds the source checkpoint")
    result = np.zeros(mapping.size, np.float32)
    result[present] = values[mapping[present]]
    return result


def extend_learner(data, mapping, *, source_count, expected_steps, batch_steps):
    """Retain Muon momentum and clocks; zero only newly introduced connections.

    The caller must verify the original policy/run/state identity before this
    transformation. This function validates the complete native snapshot format.
    """
    if len(data) < LEARNER_HEADER.size:
        raise ValueError("Truncated learner checkpoint")
    magic, epoch, steps, count, rate = LEARNER_HEADER.unpack_from(data)
    if (magic != b"METTAL01" or count != source_count
            or len(data) != LEARNER_HEADER.size + count * 4):
        raise ValueError("Learner format, parameter count or payload size differs")
    if (batch_steps <= 0 or steps != expected_steps or epoch * batch_steps != steps
            or not math.isfinite(rate) or rate < 0):
        raise ValueError("Learner counters or learning rate differ")
    momentum = np.frombuffer(data, dtype="<f4", offset=LEARNER_HEADER.size)
    extended = extend_flat(momentum, mapping)
    return LEARNER_HEADER.pack(magic, epoch, steps, extended.size, rate) + extended.astype("<f4").tobytes()

"""Explicit Classic 8192 -> 2048 game Muon resume, without editing its source."""

import hashlib
import struct

from integrations.learner_checkpoint import LEARNER_HEADER, LearnerCheckpoint


def migrate_learner(data, parameter_count, source_overrides, target_overrides):
    """Rebase only the epoch clock; preserve steps, rate and momentum bytewise.

    The pinned Muon optimizer has no epoch-based bias correction. Require
    constant learning rate and zero entropy so rebasing cannot alter schedules.
    The changed rollout changes optimization batching and needs new learning
    and throughput qualification; this is not an equivalence claim.
    """
    changed = {k for k in source_overrides.keys() | target_overrides.keys()
               if source_overrides.get(k) != target_overrides.get(k)}
    if changed != {'vec.total_agents'}:
        raise ValueError('Rollout migration permits only the parallel-game count change')
    if (source_overrides['vec.total_agents'] != 8192
            or target_overrides['vec.total_agents'] != 2048):
        raise ValueError('Only the bounded 8192-to-2048 Classic migration is supported')
    required = {'train.horizon': 256, 'train.minibatch_size': 8192,
                'train.anneal_lr': 0, 'train.ent_coef': 0.0,
                'vec.num_buffers': 1, 'train.replay_ratio': 0.5}
    if any(source_overrides.get(k) != v for k, v in required.items()):
        raise ValueError('Rollout migration requires the qualified constant-rate Classic geometry')
    state = LearnerCheckpoint.from_bytes(data, parameter_count)
    old_batch, new_batch = 8192 * 256, 2048 * 256
    if state.agent_steps <= 0 or state.epoch * old_batch != state.agent_steps:
        raise ValueError('Source learner clock does not match its rollout')
    if state.agent_steps % new_batch:
        raise ValueError('Checkpoint is not on the target rollout boundary')
    rate = struct.unpack('<f', struct.pack('<f', source_overrides['train.learning_rate']))[0]
    if state.learning_rate != rate:
        raise ValueError('Source learner learning rate differs from its run')
    epoch = state.agent_steps // new_batch
    result = bytearray(data)
    struct.pack_into('<Q', result, 8, epoch)
    result = bytes(result)
    assert result[:8] == data[:8] and result[16:] == data[16:]
    restored = LearnerCheckpoint.from_bytes(result, parameter_count)
    assert restored.epoch * new_batch == state.agent_steps
    def sha(value):
        return hashlib.sha256(value).hexdigest()
    return result, dict(recipe='classic_8192_to_2048', source_epoch=state.epoch,
                        target_epoch=epoch, agent_steps=state.agent_steps,
                        source_steps_per_epoch=old_batch, target_steps_per_epoch=new_batch,
                        source_sha256=sha(data), target_sha256=sha(result),
                        momentum_sha256=sha(data[LEARNER_HEADER.size:]),
                        learning_rate=state.learning_rate, momentum_preserved_exactly=True)

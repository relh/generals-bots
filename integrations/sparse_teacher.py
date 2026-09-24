"""Replay loss for one-hot scripted actions stored as two small metadata values."""

import jax
import jax.numpy as jnp


class SparseActionTeacher:
    def __init__(self, *, action_sizes: list[int], coefficient: float = 1.0):
        self.action_sizes = action_sizes
        self.coefficient = coefficient

    def __call__(self, policy, auxiliary, replay, *, seed):
        del auxiliary, seed
        labels = jnp.asarray(replay.metadata[..., : len(self.action_sizes)])
        if labels.shape[-1] != len(self.action_sizes):
            raise ValueError("Sparse teacher metadata does not match action heads")
        offset = 0
        loss = 0.0
        for head, size in enumerate(self.action_sizes):
            legal = jnp.asarray(replay.action_masks[..., offset : offset + size]) != 0
            logits = jnp.where(legal, policy[..., offset : offset + size], -1e9)
            log_prob = jax.nn.log_softmax(logits, axis=-1)
            active = (labels[..., head] >= 0) & (labels[..., head] < size)
            index = jnp.clip(labels[..., head].astype(jnp.int32), 0, size - 1)
            chosen = jnp.take_along_axis(log_prob, index[..., None], axis=-1)[..., 0]
            loss += -jnp.sum(jnp.where(active, chosen, 0.0)) / jnp.maximum(jnp.sum(active), 1)
            offset += size
        return self.coefficient * loss

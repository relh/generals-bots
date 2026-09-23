"""Public Generals observation and action contract for native Puffer policies."""

import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask_obs


def encode_observation(obs, *, factorized_actions=False):
    planes = obs.as_tensor().astype(jnp.float32)
    for channel in (0, 9, 10, 11, 12):
        planes = planes.at[channel].set(jnp.log1p(jnp.maximum(planes[channel], 0)) / 8.0)
    planes = planes.at[13].set(jnp.minimum(planes[13] / 1200.0, 2.0))
    moves = compute_valid_move_mask_obs(obs).transpose(2, 0, 1).reshape(-1)
    if factorized_actions:
        mask = jnp.concatenate((moves, jnp.ones((3,), dtype=bool)))
    else:
        mask = jnp.concatenate((moves, moves, jnp.ones((1,), dtype=bool)))
    return planes.reshape(-1), mask


def decode_action(index, size, split=None):
    cells = size * size
    channel, position = index // cells, index % cells
    if split is None:
        split = channel // 4
        pass_index = 8 * cells
    else:
        pass_index = 4 * cells
    move = jnp.array([0, position // size, position % size, channel % 4, split], dtype=jnp.int32)
    return jnp.where(index == pass_index, jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32), move)

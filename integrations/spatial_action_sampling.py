"""Exact flat logits for sharp route choice and exploratory full/half choice.

The public 3529-action head contains 1764 full moves, the same 1764
half moves, and pass. The transformed categorical distribution first
chooses a route from the full-move logits, then chooses its split from
the full/half pair. A single flat categorical and PPO log probability
can therefore use it without changing the native action codec.
"""

import math


MOVE_COUNT = 1764
PASS_INDEX = 3528


def public_neutral_route_bonus(observations, strength, xp):
    """Bias moves into visible empty neutral cells using only public planes."""
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 7:
        raise ValueError("Neutral route bonus requires public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    row, col = source // 21, source % 21
    delta_row = xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    delta_col = xp.take(xp.asarray((0, 0, -1, 1)), direction)
    target_row, target_col = row + delta_row, col + delta_col
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    visible_empty_neutral = (on_board &
        (xp.take(planes[..., 0, :], target, axis=-1) == 0) &
        (xp.take(planes[..., 4, :], target, axis=-1) == 0) &
        (xp.take(planes[..., 5, :], target, axis=-1) == 0) &
        (xp.take(planes[..., 6, :], target, axis=-1) == 0))
    bonus = visible_empty_neutral.astype(observations.dtype) * strength
    return xp.concatenate((bonus, bonus, xp.zeros_like(bonus[..., :1])), axis=-1)


def public_owned_split_bias(observations, strength, xp):
    """Favor half moves between owned cells from stacks of at least five."""
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 7:
        raise ValueError("Owned split bias requires public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    row, col = source // 21, source % 21
    target_row = row + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    target_col = col + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    eligible = (on_board &
                (xp.take(planes[..., 4, :], target, axis=-1) > 0.5) &
                (xp.take(planes[..., 0, :], source, axis=-1) >= math.log1p(5) / 8 - 1e-6))
    return eligible.astype(observations.dtype) * strength


def public_weak_owned_route_penalty(observations, strength, xp):
    """Discourage early shuffling of small armies between owned cells."""
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 7:
        raise ValueError("Owned route penalty requires public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    row, col = source // 21, source % 21
    target_row = row + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    target_col = col + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    early = xp.sum(planes[..., 4, :], axis=-1, keepdims=True) < 15
    weak = xp.take(planes[..., 0, :], source, axis=-1) < math.log1p(5) / 8 - 1e-6
    own = xp.take(planes[..., 4, :], target, axis=-1) > 0.5
    penalty = -(early & weak & own & on_board).astype(observations.dtype) * strength
    return xp.concatenate((penalty, penalty, xp.zeros_like(penalty[..., :1])), axis=-1)


def acting_logits(predictions, move_temperature, split_temperature, xp, split_bias=None):
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    pass_logit = predictions[..., PASS_INDEX:PASS_INDEX + 1]
    value = predictions[..., PASS_INDEX + 1:PASS_INDEX + 2]
    difference = (half - full) / split_temperature
    if split_bias is not None:
        difference = difference + split_bias
    log_partition = xp.logaddexp(0, difference)
    route = full / move_temperature
    return xp.concatenate((route - log_partition,
                           route + difference - log_partition,
                           pass_logit / move_temperature, value), axis=-1)


def raw_cotangents(predictions, logit_cotangents, value_cotangents,
                   move_temperature, split_temperature, xp, split_bias=None):
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    difference = (half - full) / split_temperature
    if split_bias is not None:
        difference = difference + split_bias
    half_probability = xp.exp(difference - xp.logaddexp(0, difference))
    full_probability = 1 - half_probability
    full_cotangent = logit_cotangents[..., :MOVE_COUNT]
    half_cotangent = logit_cotangents[..., MOVE_COUNT:PASS_INDEX]
    split_cotangent = (full_cotangent * half_probability
                       - half_cotangent * full_probability) / split_temperature
    return xp.concatenate(((full_cotangent + half_cotangent) / move_temperature + split_cotangent,
                           -split_cotangent,
                           logit_cotangents[..., PASS_INDEX:PASS_INDEX + 1] / move_temperature,
                           value_cotangents[..., None]), axis=-1)

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


def public_early_route_temperature(observations, base_temperature, early_temperature, early_turns, xp):
    """Use the public turn plane for an opening route schedule per game."""
    if observations.shape[-1] != 16 * 441:
        raise ValueError("Early route temperature requires sixteen public planes")
    if (not isinstance(base_temperature, (int, float)) or isinstance(base_temperature, bool)
            or not math.isfinite(base_temperature) or base_temperature <= 0
            or not isinstance(early_temperature, (int, float)) or isinstance(early_temperature, bool)
            or not math.isfinite(early_temperature) or early_temperature <= 0
            or isinstance(early_turns, bool) or not isinstance(early_turns, int)
            or not 0 < early_turns <= 2000):
        raise ValueError("Early route schedule requires positive temperatures and a valid turn cutoff")
    turn = xp.floor(observations[..., 11 * 441] * 2000 + 0.5)
    return xp.where(turn < early_turns, early_temperature, base_temperature)[..., None]


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


def public_safe_owned_split_bias(observations, strength, xp):
    """Favor half moves on interior owned routes from 5–19 army stacks.

    This diagnostic uses only the learner's public observation. Visible enemy
    tiles adjacent to either end of the route disable the bias.
    """
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 7:
        raise ValueError("Safe owned split bias requires public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    source_row, source_col = source // 21, source % 21
    target_row = source_row + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    target_col = source_col + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    army_log = xp.take(planes[..., 0, :], source, axis=-1)
    middle_stack = ((army_log >= math.log1p(5) / 8 - 1e-6) &
                    (army_log < math.log1p(20) / 8 - 1e-6))

    def visible_enemy_neighbor(rows, cols):
        nearby = xp.zeros_like(middle_stack, dtype=bool)
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = rows + delta_row, cols + delta_col
            valid = (row >= 0) & (row < 21) & (col >= 0) & (col < 21)
            cell = xp.clip(row, 0, 20) * 21 + xp.clip(col, 0, 20)
            nearby = nearby | (valid & (xp.take(planes[..., 5, :], cell, axis=-1) > 0.5))
        return nearby

    eligible = (on_board & middle_stack &
                (xp.take(planes[..., 4, :], source, axis=-1) > 0.5) &
                (xp.take(planes[..., 4, :], target, axis=-1) > 0.5) &
                ~visible_enemy_neighbor(source_row, source_col) &
                ~visible_enemy_neighbor(target_row, target_col))
    return eligible.astype(observations.dtype) * strength


def public_guided_owned_split_bias(observations, strength, xp):
    """Diagnostic split bias on a public route cue joining two owned stacks.

    The source has 5–19 armies and the destination has at least five. This
    only changes full versus half probability on a route; it cannot change
    the probability of choosing that route.
    """
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 11:
        raise ValueError("Guided split bias requires directional public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    row, col = source // 21, source % 21
    target_row = row + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    target_col = col + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    source_army = xp.take(planes[..., 0, :], source, axis=-1)
    target_army = xp.take(planes[..., 0, :], target, axis=-1)
    directional_cue = xp.concatenate(tuple(planes[..., 7 + i, :] for i in range(4)), axis=-1)
    eligible = (on_board &
                (source_army >= math.log1p(5) / 8 - 1e-6) &
                (source_army < math.log1p(20) / 8 - 1e-6) &
                (target_army >= math.log1p(5) / 8 - 1e-6) &
                (xp.take(planes[..., 4, :], target, axis=-1) > .5) &
                (directional_cue > .5))
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


def public_doomed_attack_route_penalty(observations, strength, xp):
    """Discourage attacks on visible enemies that a full move cannot capture."""
    if observations.shape[-1] % 441 or observations.shape[-1] // 441 < 7:
        raise ValueError("Attack route penalty requires public 21x21 observations")
    planes = observations.reshape((*observations.shape[:-1], -1, 441))
    routes = xp.arange(MOVE_COUNT)
    source = routes % 441
    direction = routes // 441
    row, col = source // 21, source % 21
    target_row = row + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    target_col = col + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target = xp.clip(target_row, 0, 20) * 21 + xp.clip(target_col, 0, 20)
    source_army = xp.floor(xp.expm1(xp.take(planes[..., 0, :], source, axis=-1) * 8) + .5)
    target_army = xp.floor(xp.expm1(xp.take(planes[..., 0, :], target, axis=-1) * 8) + .5)
    doomed = (on_board &
              (xp.take(planes[..., 5, :], target, axis=-1) > .5) &
              (source_army - 1 <= target_army))
    penalty = -doomed.astype(observations.dtype) * strength
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

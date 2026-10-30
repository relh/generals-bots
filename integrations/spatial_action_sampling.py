"""Exact flat logits for sharp route choice and exploratory full/half choice.

The public 3529-action head contains 1764 full moves, the same 1764
half moves, and pass. The transformed categorical distribution first
chooses a route from a weighted full/half score, then chooses its split
from the full/half pair. Weight zero preserves the original full-only
route distribution. A single flat categorical and PPO log probability
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


def validate_full_action_temperature(temperature):
    """A static multiplier applied after every action-logit adjustment."""
    if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Full action temperature must be finite and positive")
    return temperature


def scale_action_logits(logits, temperature):
    """Scale actions only; callers must exclude values and apply all priors first."""
    validate_full_action_temperature(temperature)
    return logits if temperature == 1 else logits / temperature


def acting_logits(predictions, move_temperature, split_temperature, xp,
                  *, route_half_weight=0.0):
    if not isinstance(route_half_weight, (int, float)) or isinstance(route_half_weight, bool) or (
            not math.isfinite(route_half_weight) or not 0 <= route_half_weight <= 1):
        raise ValueError("Route half weight must be finite and between zero and one")
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    pass_logit = predictions[..., PASS_INDEX:PASS_INDEX + 1]
    value = predictions[..., PASS_INDEX + 1:PASS_INDEX + 2]
    difference = (half - full) / split_temperature
    log_partition = xp.logaddexp(0, difference)
    route = (full + route_half_weight * (half - full)) / move_temperature
    return xp.concatenate((route - log_partition,
                           route + difference - log_partition,
                           pass_logit / move_temperature, value), axis=-1)


def raw_cotangents(predictions, logit_cotangents, value_cotangents,
                   move_temperature, split_temperature, xp,
                   *, route_half_weight=0.0):
    if not isinstance(route_half_weight, (int, float)) or isinstance(route_half_weight, bool) or (
            not math.isfinite(route_half_weight) or not 0 <= route_half_weight <= 1):
        raise ValueError("Route half weight must be finite and between zero and one")
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    difference = (half - full) / split_temperature
    half_probability = xp.exp(difference - xp.logaddexp(0, difference))
    full_probability = 1 - half_probability
    full_cotangent = logit_cotangents[..., :MOVE_COUNT]
    half_cotangent = logit_cotangents[..., MOVE_COUNT:PASS_INDEX]
    split_cotangent = (full_cotangent * half_probability
                       - half_cotangent * full_probability) / split_temperature
    route_cotangent = full_cotangent + half_cotangent
    return xp.concatenate(((1 - route_half_weight) * route_cotangent / move_temperature + split_cotangent,
                           route_half_weight * route_cotangent / move_temperature - split_cotangent,
                           logit_cotangents[..., PASS_INDEX:PASS_INDEX + 1] / move_temperature,
                           value_cotangents[..., None]), axis=-1)

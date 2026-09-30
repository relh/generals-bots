"""Exact flat logits for sharp route choice and exploratory full/half choice.

The public 3529-action head contains 1764 full moves, the same 1764
half moves, and pass. The transformed categorical distribution first
chooses a route from the full-move logits, then chooses its split from
the full/half pair. A single flat categorical and PPO log probability
can therefore use it without changing the native action codec.
"""


MOVE_COUNT = 1764
PASS_INDEX = 3528


def acting_logits(predictions, move_temperature, split_temperature, xp):
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    pass_logit = predictions[..., PASS_INDEX:PASS_INDEX + 1]
    value = predictions[..., PASS_INDEX + 1:PASS_INDEX + 2]
    difference = (half - full) / split_temperature
    log_partition = xp.logaddexp(0, difference)
    route = full / move_temperature
    return xp.concatenate((route - log_partition,
                           route + difference - log_partition,
                           pass_logit / move_temperature, value), axis=-1)


def raw_cotangents(predictions, logit_cotangents, value_cotangents,
                   move_temperature, split_temperature, xp):
    full = predictions[..., :MOVE_COUNT]
    half = predictions[..., MOVE_COUNT:PASS_INDEX]
    difference = (half - full) / split_temperature
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

"""Public-mask reconstruction and differentiable action-tail exploration.

These primitives alone do not enable a sampler. Training, export and serving
must bind the same scale before a policy using them can be qualified.
"""

import math


def public_action_mask(observations, xp):
    """Reconstruct Classic flat actions from the sixteen public feature planes.

    An owned source sees its adjacent destinations. Thus plane three's union
    of mountains and hidden structures is unambiguous on every movable route.
    Padded cells adjacent to ownership are encoded as visible mountains.
    """
    if not hasattr(observations, "shape") or not observations.shape or observations.shape[-1] != 16 * 441:
        raise ValueError("Public action mask requires sixteen 21x21 planes")
    planes = observations.reshape((*observations.shape[:-1], 16, 441))
    routes = xp.arange(1764)
    source, direction = routes % 441, routes // 441
    row = source // 21 + xp.take(xp.asarray((-1, 1, 0, 0)), direction)
    col = source % 21 + xp.take(xp.asarray((0, 0, -1, 1)), direction)
    on_board = (row >= 0) & (row < 21) & (col >= 0) & (col < 21)
    target = xp.clip(row, 0, 20) * 21 + xp.clip(col, 0, 20)
    own = xp.take(planes[..., 4, :], source, axis=-1) > .5
    enough_army = xp.take(planes[..., 0, :], source, axis=-1) >= math.log1p(2) / 8 - 1e-6
    blocked = xp.take(planes[..., 3, :], target, axis=-1) > .5
    moves = on_board & own & enough_army & ~blocked
    return xp.concatenate((moves, moves, xp.ones_like(moves[..., :1])), axis=-1)


def validate_log_gap_scale(scale):
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or not math.isfinite(scale) or scale < 0):
        raise ValueError("Log gap scale must be finite and nonnegative; zero disables it")
    return scale


def log_gap_logits(logits, legal, scale, xp):
    """Preserve legal rankings while replacing exponential tails with power tails.

    Operates on actions only, after all priors and temperature adjustments.
    Invalid actions receive finite placeholders and must still be masked by
    the categorical sampler. Scale zero preserves legacy logits exactly.
    """
    validate_log_gap_scale(scale)
    if logits.shape != legal.shape:
        raise ValueError("Exploration logits and public mask shapes differ")
    if scale == 0:
        return logits
    maximum = xp.max(xp.where(legal, logits, -xp.inf), axis=-1, keepdims=True)
    gap = xp.where(legal, maximum - logits, 0)
    return -scale * xp.log1p(gap / scale)


def log_gap_cotangents(logits, legal, cotangents, scale, xp):
    """Exact VJP including the legal maximum; tied maxima share its derivative."""
    validate_log_gap_scale(scale)
    if logits.shape != legal.shape or logits.shape != cotangents.shape:
        raise ValueError("Exploration cotangents, logits and public mask shapes differ")
    if scale == 0:
        return cotangents
    maximum = xp.max(xp.where(legal, logits, -xp.inf), axis=-1, keepdims=True)
    gap = xp.where(legal, maximum - logits, 0)
    weighted = xp.where(legal, cotangents * scale / (scale + gap), 0)
    tied = legal & (logits == maximum)
    count = xp.maximum(xp.sum(tied, axis=-1, keepdims=True), 1)
    return weighted - tied * xp.sum(weighted, axis=-1, keepdims=True) / count

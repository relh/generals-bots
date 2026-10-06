"""Experimental bounded force concentration potential; not wired into PPO."""


def concentration(armies, ownership, xp):
    """Per-seat squared stack share; never square the integer army dtype."""
    forces = armies.astype(xp.float32)[..., None, :, :] * ownership
    totals = forces.sum(axis=(-2, -1))
    return (forces * forces).sum(axis=(-2, -1)) / (totals * totals + 1.)


def potential(armies, ownership, time, xp):
    scores = concentration(armies, ownership, xp)
    return (scores - scores[..., ::-1]) * (xp.asarray(time) >= 100)[..., None]


def shaping(previous, following, done, *, gamma=.999, weight=.05):
    """Unscaled addition inside the current potential; terminal potential is zero."""
    return weight * (gamma * following * (1 - done) - previous)

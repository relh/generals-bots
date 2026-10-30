"""Explicit entropy-only experiment support for the pinned Puffer resume guard."""

import json
import math


def entropy_resume_overrides_compatible(before, after):
    """Allow one finite entropy coefficient change; retain every other override."""
    key = "train.ent_coef"
    if key not in before or key not in after:
        return False
    for value in (before[key], after[key]):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value) or value < 0:
            return False
    if {k: v for k, v in before.items() if k != key} != {
        k: v for k, v in after.items() if k != key
    }:
        return False
    print("ENTROPY_RESUME_OVERRIDE " + json.dumps({
        "key": key, "source": before[key], "target": after[key],
        "scope": "policy and optimizer retained; all other resume checks retained",
    }), flush=True)
    return True


def entropy_resume_source(source):
    """Replace exactly the overrides equality guard, leaving seed checks intact."""
    old = "if source.config.overrides != config.overrides or source.config.seed != config.seed:"
    new = (
        "if (source.config.overrides != config.overrides and not "
        "entropy_resume_overrides_compatible(source.config.overrides, config.overrides)) "
        "or source.config.seed != config.seed:"
    )
    if source.count(old) != 1:
        raise ValueError("Pinned learner resume guard differs from its expected source")
    return source.replace(old, new)

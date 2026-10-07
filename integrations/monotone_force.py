"""Fixed isolated consolidation hypothesis; no sampler or observation changes."""

SOURCE_POLICY = "f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14"
TRANSFER = "monotone-force-k100-v1"


def potential(armies, ownership, time, xp):
    forces = armies.astype(xp.float32)[..., None, :, :] * ownership
    squares = (forces * forces).sum(axis=(-2, -1))
    score = squares / (10000.0 + squares)
    return (score - score[..., ::-1]) * (xp.asarray(time) >= 100)[..., None]


def validate_transfer(metadata, objective, declaration, restore_learner):
    """One declared fresh-optimizer source-to-potential transfer; no source rewrite."""
    original = metadata["training_contract"]
    changed = dict(objective, schema="generals-classic-training-contract-v1")
    if (declaration != TRANSFER or restore_learner
            or metadata["policy_sha256"] != SOURCE_POLICY
            or original != changed
            or objective["schema"] != "generals-classic-monotone-force-v1"
            or objective["learner_gamma"] != .999
            or objective["reward"] != dict(terminal_reward_mode="win_only", reward_scale=.5,
                 shaping_weight=.25, shaping_gamma=.999, army_shaping_weight=.5, land_shaping_weight=.3)):
        raise ValueError("Monotone force transfer differs from the preregistered source/objective/fresh optimizer")

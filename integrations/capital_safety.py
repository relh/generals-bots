"""Public one-turn Classic capital reinforcement and guaranteed countercapture."""


def enabled(environ):
    value = environ.get("METTA_SPATIAL_CAPITAL_SAFETY", "False")
    if value not in ("True", "False"):
        raise ValueError("Capital safety environment setting must be True or False")
    return value == "True"


def safe_actions(observations, xp):
    """Return a preference mask; all actions remain unchanged outside proven cases.

    Army recovery is exact for the bounded integer range tested below 16,384.
    Larger adjacent armies disable the constraint. The official legality mask
    still applies separately; this function never grants a game action.
    """
    from integrations.spatial_exploration import public_action_mask

    if observations.shape[-1:] != (7056,):
        raise ValueError("Capital safety requires canonical public observations")
    planes = observations.reshape(*observations.shape[:-1], 16, 441)
    army = xp.floor(xp.expm1(planes[..., 0, :] * 8) + .5)
    own, enemy = planes[..., 4, :] > .5, planes[..., 5, :] > .5
    general = planes[..., 1, :] > .5
    capital = own & general
    sites = xp.arange(441)
    source = xp.broadcast_to(sites, (4, 441))
    destination = source + xp.asarray((-21, 21, -1, 1))[:, None]
    valid = (destination >= 0) & (destination < 441)
    valid = valid & xp.stack((sites >= 21, sites < 420, sites % 21 > 0, sites % 21 < 20))
    destination = xp.clip(destination, 0, 440)
    into_capital = capital[..., destination] & valid
    threats = xp.where(enemy[..., None, :] & into_capital, army[..., None, :], 0)
    strongest = threats.max(axis=(-2, -1))
    garrison = xp.where(capital, army, 0).sum(axis=-1)
    full = xp.maximum(army - 1, 0)
    moved = xp.stack((full, xp.floor(army / 2)), axis=-2)
    reinforces = (own[..., None, None, :] & into_capital[..., None, :, :]
                  & (garrison[..., None, None, None] + moved[..., :, None, :]
                     >= strongest[..., None, None, None] - 1))
    # An attack on a visible enemy general wins before any capital attack if
    # its source stack is strictly larger, and its source is not our capital.
    wins = (own[..., None, None, :] & ~capital[..., None, None, :]
            & (enemy & general)[..., None, destination] & valid
            & (moved[..., :, None, :] > army[..., None, destination])
            & (army[..., None, None, :] > strongest[..., None, None, None]))
    selected = (reinforces | wins).reshape(*observations.shape[:-1], 3528)
    selected = xp.concatenate((selected, xp.zeros(selected.shape[:-1] + (1,), dtype=bool)), axis=-1)
    legal = public_action_mask(observations, xp)
    bounded = xp.all(xp.where(capital | (threats > 0).any(axis=-2), army <= 16384, True), axis=-1)
    # Bound every candidate reinforcement/source participating in the proof.
    bounded = bounded & xp.all(xp.where(selected[..., :3528].reshape(*observations.shape[:-1], 2, 4, 441).any(axis=(-3, -2)), army <= 16384, True), axis=-1)
    active = (capital.sum(axis=-1) == 1) & (strongest - 1 > garrison) & bounded & (selected & legal).any(axis=-1)
    return xp.where(active[..., None], selected, xp.ones_like(legal))


def constrain_logits(logits, observations, xp):
    from integrations.spatial_exploration import public_action_mask

    if logits.shape != observations.shape[:-1] + (3529,):
        raise ValueError("Capital safety logits and public observations differ")
    allowed = safe_actions(observations, xp)
    active = (~allowed).any(axis=-1)
    # Center first so any finite safe logit outranks the finite native sentinel.
    candidates = xp.where(allowed & public_action_mask(observations, xp), logits, -xp.inf)
    index = candidates.argmax(axis=-1)[..., None]
    reference = xp.take_along_axis(logits, index, axis=-1)
    guarded = xp.where(allowed, logits - reference, xp.asarray(-1e9, dtype=logits.dtype))
    return xp.where(active[..., None], guarded, logits)


def install_sampler(source):
    """Keep native CDF endpoint fallback inside positive probability support."""
    path = source / "src/pufferl.cu"
    text = path.read_text()
    replacements = (
        ("    bool norm_adv;\n", "    bool norm_adv;\n    bool capital_safety;\n"),
        ('        .norm_adv = puf_ini_get(ini, "train", "norm_adv") != 0,',
         '        .norm_adv = puf_ini_get(ini, "train", "norm_adv") != 0,\n'
         '        .capital_safety = getenv("METTA_SPATIAL_CAPITAL_SAFETY") != NULL &&\n'
         '            strcmp(getenv("METTA_SPATIAL_CAPITAL_SAFETY"), "True") == 0,'),
        ("        int mask_stride) {", "        int mask_stride, bool capital_safety) {"),
        ("            mask_b.data, mask_stride);", "            mask_b.data, mask_stride, hypers->capital_safety);"),
        ('                    if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f) {',
         '                    if (to_float(action_mask[mask_base + logits_offset + a]) != 0.0f &&\n'
         '                        (!capital_safety || expf(cache[a] - logsumexp) > 0.0f)) {'),
    )
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Capital sampler anchor changed: {old}")
        text = text.replace(old, new, 1)
    path.write_text(text)

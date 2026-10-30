"""Post-action destination categories for held-out Classic evaluation."""

import jax
import jax.numpy as jnp


@jax.jit
def destination_categories(ownership_neutral, ownership, sides, actions, active):
    """Return 0 inactive/invalid, 1 neutral, 2 owned, 3 enemy, 4 pass.

    Ownership is the pre-action game state. These omniscient labels are
    diagnostics only and must never be supplied to the acting policy.
    """
    rows = jnp.arange(actions.shape[0])
    moving = actions < 3528
    route = actions % 1764
    direction, source = jnp.divmod(route, 441)
    source_row, source_col = jnp.divmod(source, 21)
    target_row = source_row + jnp.take(jnp.asarray((-1, 1, 0, 0)), direction)
    target_col = source_col + jnp.take(jnp.asarray((0, 0, -1, 1)), direction)
    valid = (target_row >= 0) & (target_row < 21) & (target_col >= 0) & (target_col < 21)
    target_row = jnp.clip(target_row, 0, 20)
    target_col = jnp.clip(target_col, 0, 20)
    neutral = ownership_neutral[rows, target_row, target_col]
    owned = ownership[rows, sides, target_row, target_col]
    enemy = ownership[rows, 1 - sides, target_row, target_col]
    destination = jnp.where(neutral, 1, jnp.where(owned, 2, jnp.where(enemy, 3, 0)))
    return jnp.where(active, jnp.where(moving, jnp.where(valid, destination, 0), 4), 0).astype(jnp.int32)

"""Pure JAX sampling shared by frozen opponents and standalone parity audits."""

import jax
import jax.numpy as jnp

from integrations.spatial_action_sampling import (
    acting_logits,
    scale_action_logits,
    public_doomed_attack_route_penalty,
    public_early_route_temperature,
    public_neutral_route_bonus,
    public_weak_owned_route_penalty,
)


@jax.jit
def sample_flat_logits(key, logits, legal):
    return jax.random.categorical(key, jnp.where(legal, logits, -jnp.inf), axis=-1)


def frozen_action_indices(policy, outputs, masks, keys, observations=None):
    """Select frozen opponent actions using the bundle's serving contract."""
    if policy.action_mode != "structured_sample":
        raise ValueError("Frozen opponents require the current structured sampler")
    move_temperature = policy.move_temperature
    if policy.early_route_temperature is not None:
        if observations is None:
            raise ValueError("Early route schedule requires frozen public observations")
        move_temperature = public_early_route_temperature(
            observations, move_temperature, policy.early_route_temperature, policy.early_route_turns, jnp)
    logits = acting_logits(outputs, move_temperature, policy.split_temperature, jnp,
                           route_half_weight=policy.route_half_weight)[:, :3529]
    if policy.neutral_route_bias:
        if observations is None:
            raise ValueError("Neutral route bias requires frozen public observations")
        logits += public_neutral_route_bonus(observations, policy.neutral_route_bias, jnp)
    if policy.weak_owned_route_penalty:
        if observations is None:
            raise ValueError("Weak owned route penalty requires frozen public observations")
        logits += public_weak_owned_route_penalty(observations, policy.weak_owned_route_penalty, jnp)
    if policy.doomed_attack_route_penalty:
        if observations is None:
            raise ValueError("Doomed attack route penalty requires frozen public observations")
        logits += public_doomed_attack_route_penalty(observations, policy.doomed_attack_route_penalty, jnp)
    logits = scale_action_logits(logits, policy.full_action_temperature)
    if policy.log_gap_scale:
        from integrations.spatial_exploration import log_gap_logits, public_action_mask

        logits = log_gap_logits(logits, public_action_mask(observations, jnp), policy.log_gap_scale, jnp)
    if policy.capital_safety:
        from integrations.capital_safety import constrain_logits

        logits = constrain_logits(logits, observations, jnp)
    legal_logits = jnp.where(masks, logits, -jnp.inf)
    random_keys = jax.vmap(lambda key: jax.random.fold_in(key, 834))(keys)
    return jax.vmap(jax.random.categorical)(random_keys, legal_logits).astype(jnp.int32)

"""Pure JAX sampling shared by frozen opponents and standalone parity audits."""

import jax
import jax.numpy as jnp

from integrations.spatial_action_sampling import (
    acting_logits,
    public_doomed_attack_route_penalty,
    public_early_route_temperature,
    public_neutral_route_bonus,
    public_weak_owned_route_penalty,
)


def frozen_action_indices(policy, outputs, masks, keys, observations=None):
    """Select frozen opponent actions using the bundle's serving contract."""
    if policy.action_mode == "structured_sample":
        move_temperature = policy.move_temperature
        if getattr(policy, "early_route_temperature", None) is not None:
            if observations is None:
                raise ValueError("Early route schedule requires frozen public observations")
            move_temperature = public_early_route_temperature(
                observations, move_temperature, policy.early_route_temperature, policy.early_route_turns, jnp)
        logits = acting_logits(outputs, move_temperature, policy.split_temperature, jnp,
                               route_half_weight=getattr(policy, "route_half_weight", 0.0))[:, :3529]
        if getattr(policy, "neutral_route_bias", 0.0):
            if observations is None:
                raise ValueError("Neutral route bias requires frozen public observations")
            logits += public_neutral_route_bonus(observations, policy.neutral_route_bias, jnp)
        if getattr(policy, "weak_owned_route_penalty", 0.0):
            if observations is None:
                raise ValueError("Weak owned route penalty requires frozen public observations")
            logits += public_weak_owned_route_penalty(observations, policy.weak_owned_route_penalty, jnp)
        if getattr(policy, "doomed_attack_route_penalty", 0.0):
            if observations is None:
                raise ValueError("Doomed attack route penalty requires frozen public observations")
            logits += public_doomed_attack_route_penalty(observations, policy.doomed_attack_route_penalty, jnp)
        legal_logits = jnp.where(masks, logits, -jnp.inf)
        random_keys = jax.vmap(lambda key: jax.random.fold_in(key, 834))(keys)
        return jax.vmap(jax.random.categorical)(random_keys, legal_logits).astype(jnp.int32)
    return jnp.argmax(jnp.where(masks, outputs[:, :3529], -jnp.inf), axis=1).astype(jnp.int32)


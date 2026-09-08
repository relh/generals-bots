"""PPO math: explicit next values, timeout-aware GAE, minibatch diagnostics."""
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class Batch(NamedTuple):
    observations: jax.Array
    masks: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    advantages: jax.Array
    returns: jax.Array


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    """Bootstrap time limits from last_state, but stop traces at every reset."""
    deltas = rewards + gamma * jnp.where(terminated, 0, next_values) - values

    def step(carry, xs):
        delta, boundary = xs
        advantage = delta + gamma * lam * jnp.where(boundary, 0, carry)
        return advantage, advantage

    return jax.lax.scan(step, jnp.zeros_like(values[0]),
                        (deltas, terminated | truncated), reverse=True)[1]


def loss(network, batch, clip=0.2, entropy_weight=0.01, value_weight=0.5):
    logits, values = jax.vmap(network)(batch.observations, batch.masks)
    logs = jax.nn.log_softmax(logits)
    new_logs = jnp.take_along_axis(logs, batch.actions[:, None], axis=1)[:, 0]
    logratio = new_logs - batch.logprobs
    ratio = jnp.exp(logratio)
    policy = -jnp.minimum(ratio * batch.advantages,
                         jnp.clip(ratio, 1 - clip, 1 + clip) * batch.advantages).mean()
    value = 0.5 * jnp.square(values - batch.returns).mean()
    entropy = -(jax.nn.softmax(logits) * logs).sum(-1).mean()
    diagnostics = dict(policy_loss=policy, value_loss=value, entropy=entropy,
                       approx_kl=(ratio - 1 - logratio).mean(),
                       clip_fraction=(jnp.abs(ratio - 1) > clip).mean())
    return policy + value_weight * value - entropy_weight * entropy, diagnostics


@eqx.filter_jit
def update(network, opt_state, batch, optimizer, clip=0.2, entropy_weight=0.01):
    (value, diagnostics), gradients = eqx.filter_value_and_grad(loss, has_aux=True)(
        network, batch, clip, entropy_weight)
    updates, opt_state = optimizer.update(gradients, opt_state, eqx.filter(network, eqx.is_array))
    diagnostics = {**diagnostics, "loss": value, "grad_norm": optax.global_norm(gradients)}
    return eqx.apply_updates(network, updates), opt_state, diagnostics

"""Compiled PPO minibatch driver with the reference loop's stopping semantics."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .ppo import update

DIAGNOSTICS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "loss", "grad_norm")


class OptimizationResult(NamedTuple):
    network: object
    optimizer_state: object
    key: jax.Array
    update_count: jax.Array
    diagnostic_count: jax.Array
    kl_early_stop: jax.Array
    nonfinite: jax.Array
    diagnostics: dict


@eqx.filter_jit
def optimize(
    network,
    opt_state,
    batch,
    optimizer,
    key,
    *,
    epochs,
    minibatch_size,
    clip=0.2,
    entropy_weight=0.01,
    target_kl=0.03,
    update_fn=update,
):
    """Run updates until all epochs finish, KL exceeds its limit, or failure.

    Split the RNG only when entering an epoch. KL is checked AFTER applying the
    update whose pre-update loss produced it. A nonfinite update also applies,
    matching the reference, but no subsequent update runs. The caller must call
    ``summarize`` to raise on failure before logging/saving a successful iteration.

    Buffers include one row per potential update; only ``diagnostic_count`` rows
    are valid for reporting. Failed-update diagnostics are stored at the next row.
    ``update_fn`` is injectable for control-flow regression tests.
    """
    n = batch.observations.shape[0]
    if n < 1 or epochs < 1 or minibatch_size < 1:
        raise ValueError("batch size, epochs and minibatch size must be positive")
    full_size = min(minibatch_size, n)
    batches_per_epoch = (n + full_size - 1) // full_size
    remainder = n % full_size
    max_updates = epochs * batches_per_epoch
    parameters, static_network = eqx.partition(network, eqx.is_array)
    history = {name: jnp.zeros(max_updates, jnp.float32) for name in DIAGNOSTICS}
    initial = (
        parameters,
        opt_state,
        key,
        jnp.arange(n),
        jnp.int32(0),
        jnp.int32(0),
        jnp.array(False),
        jnp.array(False),
        history,
    )

    def condition(carry):
        return (carry[4] < max_updates) & ~carry[6] & ~carry[7]

    def body(carry):
        parameters, opt_state, key, indices, count, valid_count, _, _, history = carry
        slot = count % batches_per_epoch

        def shuffle(args):
            key, _ = args
            key, shuffle_key = jax.random.split(key)
            return key, jax.random.permutation(shuffle_key, n)

        key, indices = jax.lax.cond(slot == 0, shuffle, lambda args: args, (key, indices))
        current_network = eqx.combine(parameters, static_network)

        def apply(size):
            selected = jax.lax.dynamic_slice_in_dim(indices, slot * full_size, size)
            minibatch = jax.tree.map(lambda values: values[selected], batch)
            new_network, new_opt_state, diagnostics = update_fn(
                current_network, opt_state, minibatch, optimizer, clip, entropy_weight
            )
            new_parameters = eqx.filter(new_network, eqx.is_array)
            return new_parameters, new_opt_state, diagnostics

        if remainder:
            parameters, opt_state, diagnostics = jax.lax.cond(
                slot == batches_per_epoch - 1, lambda: apply(remainder), lambda: apply(full_size)
            )
        else:
            parameters, opt_state, diagnostics = apply(full_size)
        finite = jnp.all(jnp.stack([jnp.isfinite(diagnostics[name]) for name in DIAGNOSTICS]))
        history = {name: values.at[count].set(diagnostics[name]) for name, values in history.items()}
        stop_early = finite & (diagnostics["approx_kl"] > target_kl)
        return (
            parameters,
            opt_state,
            key,
            indices,
            count + 1,
            valid_count + finite.astype(jnp.int32),
            stop_early,
            ~finite,
            history,
        )

    parameters, opt_state, key, _, count, valid_count, early, failed, history = jax.lax.while_loop(
        condition, body, initial
    )
    return OptimizationResult(
        eqx.combine(parameters, static_network), opt_state, key, count, valid_count, early, failed, history
    )


def summarize(result):
    """Transfer diagnostics together and preserve the reference float64 means."""
    count, failed, early, history = jax.device_get(
        (result.diagnostic_count, result.nonfinite, result.kl_early_stop, result.diagnostics)
    )
    count = int(count)
    if failed:
        failure = {name: float(values[count]) for name, values in history.items()}
        raise FloatingPointError(f"nonfinite optimizer diagnostic: {failure}")
    means = {name: float(np.mean(values[:count], dtype=np.float64)) for name, values in history.items()}
    return means, count, bool(early)

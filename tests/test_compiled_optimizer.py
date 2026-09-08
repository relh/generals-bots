"""Compiled PPO scheduling must match reference updates, failures and RNG use."""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from generals.core import game
from generals.training.checkpoint import load_checkpoint, save_checkpoint
from generals.training.network import SpatialPolicy, action_mask, features
from generals.training.optimization import DIAGNOSTICS, optimize, summarize
from generals.training.ppo import Batch, update


def reference(
    network,
    state,
    batch,
    optimizer,
    key,
    *,
    epochs,
    minibatch_size,
    target_kl=0.03,
    clip=0.2,
    entropy_weight=0.01,
    update_fn=update,
):
    rows = []
    attempted = 0
    failed = early = False
    for _ in range(epochs):
        key, shuffle = jax.random.split(key)
        indices = jax.random.permutation(shuffle, len(batch.observations))
        for start in range(0, len(indices), minibatch_size):
            minibatch = jax.tree.map(lambda x: x[indices[start : start + minibatch_size]], batch)
            network, state, diagnostics = update_fn(network, state, minibatch, optimizer, clip, entropy_weight)
            attempted += 1
            row = {name: float(value) for name, value in diagnostics.items()}
            if not all(np.isfinite(value) for value in row.values()):
                failed = True
                break
            rows.append(row)
            if row["approx_kl"] > target_kl:
                early = True
                break
        if failed or early:
            break
    return network, state, key, rows, attempted, failed, early


class ToyNetwork(eqx.Module):
    weight: jax.Array


@eqx.filter_jit
def toy_update(network, state, batch, optimizer, clip, entropy_weight, *, fail_at=-1):
    del optimizer, clip, entropy_weight
    state = state + 1
    value = batch.observations.mean() * state
    network = ToyNetwork(network.weight * 1.125 + value)
    diagnostics = {name: value + i for i, name in enumerate(DIAGNOSTICS)}
    diagnostics["approx_kl"] = state.astype(jnp.float32)
    diagnostics["loss"] = jnp.where(state == fail_at, jnp.nan, diagnostics["loss"])
    return network, state, diagnostics


@pytest.mark.parametrize(
    "n,minibatch,threshold,fail_at",
    [
        (10, 4, 100, -1),  # Partial final minibatch across three epochs.
        (3, 8, 100, -1),  # Batch smaller than requested minibatch.
        (10, 4, 0.5, -1),  # First update applies before KL stopping.
        (10, 4, 2.5, -1),  # Final minibatch of first epoch.
        (10, 4, 4.5, -1),  # Mid-second-epoch stop, exactly two RNG splits.
        (10, 4, 2.0, -1),  # Equality must not trigger strict-greater-than check.
        (10, 4, 100, 1),  # Failed first update applies, zero valid diagnostics.
        (10, 4, 100, 5),  # Failure after entering second epoch.
    ],
)
def test_schedule_statistics_rng_and_failure_semantics(n, minibatch, threshold, fail_at):
    batch = Batch(*(jnp.arange(n, dtype=jnp.float32) for _ in range(6)))
    args = (ToyNetwork(jnp.float32(0)), jnp.int32(0), batch, None, jax.random.PRNGKey(19))
    options = dict(
        epochs=3, minibatch_size=minibatch, target_kl=threshold, update_fn=partial(toy_update, fail_at=fail_at)
    )
    expected = reference(*args, **options)
    actual = optimize(*args, **options)
    np.testing.assert_array_equal(actual.network.weight, expected[0].weight)
    np.testing.assert_array_equal(actual.optimizer_state, expected[1])
    np.testing.assert_array_equal(actual.key, expected[2])
    assert int(actual.update_count) == expected[4]
    assert int(actual.diagnostic_count) == len(expected[3])
    assert bool(actual.nonfinite) == expected[5]
    assert bool(actual.kl_early_stop) == expected[6]
    for name, buffer in actual.diagnostics.items():
        np.testing.assert_array_equal(buffer[: len(expected[3])], [row[name] for row in expected[3]])
        np.testing.assert_array_equal(buffer[expected[4] :], 0)
    if expected[5]:
        with pytest.raises(FloatingPointError, match="nonfinite optimizer diagnostic"):
            summarize(actual)
    else:
        means, count, early = summarize(actual)
        assert count == len(expected[3]) and early == expected[6]
        for name, mean in means.items():
            assert mean == np.mean([row[name] for row in expected[3]])


def fixture():
    network = SpatialPolicy(jax.random.PRNGKey(0), width=4)
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(0.001))
    state = optimizer.init(eqx.filter(network, eqx.is_array))
    grid = jnp.zeros((4, 4), jnp.int32).at[0, 0].set(1).at[3, 3].set(2)
    board = game.create_initial_state(grid)
    board = board._replace(armies=board.armies.at[0, 0].set(5))
    obs = game.get_observation(board, 0)
    inputs, mask = features(obs), action_mask(obs)
    n = 10
    logits, _ = network(inputs, mask)
    action = jnp.argmax(logits)
    logprob = jax.nn.log_softmax(logits)[action]
    batch = Batch(
        jnp.broadcast_to(inputs, (n,) + inputs.shape),
        jnp.broadcast_to(mask, (n,) + mask.shape),
        jnp.full(n, action),
        jnp.full(n, logprob),
        jnp.linspace(-1, 1, n),
        jnp.linspace(0, 1, n),
    )
    return network, state, batch, optimizer, jax.random.PRNGKey(13)


def test_real_ppo_parameters_optimizer_and_diagnostics_match_reference():
    args = fixture()
    options = dict(epochs=2, minibatch_size=4, target_kl=100.0)
    expected = reference(*args, **options)
    actual = optimize(*args, **options)
    for left, right in zip(jax.tree.leaves((actual.network, actual.optimizer_state)), jax.tree.leaves(expected[:2])):
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(actual.key, expected[2])
    means, count, early = summarize(actual)
    assert count == 6 and not early
    for name, value in means.items():
        np.testing.assert_allclose(value, np.mean([row[name] for row in expected[3]]), atol=2e-6, rtol=2e-5)


def test_compiled_optimizer_checkpoint_resume_is_exact(tmp_path):
    network, state, batch, optimizer, key = fixture()
    options = dict(epochs=2, minibatch_size=4, target_kl=100.0)
    first = optimize(network, state, batch, optimizer, key, **options)
    path = tmp_path / "optimizer.pkl"
    save_checkpoint(path, dict(network=first.network, optimizer=first.optimizer_state, rng=first.key))
    loaded = load_checkpoint(path)
    expected = optimize(first.network, first.optimizer_state, batch, optimizer, first.key, **options)
    actual = optimize(loaded["network"], loaded["optimizer"], batch, optimizer, loaded["rng"], **options)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left, right)


def test_full_trainer_compiled_driver_resume_is_exact(tmp_path, monkeypatch):
    from dataclasses import replace

    from generals.training import train

    # Exercise GPU's selected driver using small CPU inputs, without taking GPU resources.
    monkeypatch.setattr(train, "optimizer_implementation", lambda: "compiled-v1")
    config = train.Config(
        num_envs=2,
        steps=4,
        iterations=2,
        pool_size=2,
        board_size=6,
        opponents="random",
        width=4,
        minibatch_size=3,
        epochs=2,
        save_every=1,
        truncation=5,
    )
    whole = train.run(config, tmp_path / "whole")
    train.run(replace(config, iterations=1), tmp_path / "split")
    resumed = train.run(config, tmp_path / "split", tmp_path / "split/checkpoint.pkl")
    assert whole["metadata"]["optimizer_implementation"] == "compiled-v1"
    for left, right in zip(jax.tree.leaves(whole), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(left, right)

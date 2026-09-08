"""Numerical regression checks for the experimental PPO trainer."""
import pytest

pytest.importorskip("equinox")
pytest.importorskip("optax")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .network import PolicyValueNetwork
from .train import compute_gae, train_step


def test_gae_stops_at_episode_boundary():
    # A large value in the following episode must not leak across the reset.
    advantages = compute_gae(
        jnp.array([[1.0], [2.0], [3.0]]),
        jnp.array([[0.5], [0.7], [100.0]]),
        jnp.array([[False], [True], [False]]),
        jnp.array([4.0]), gamma=1.0, lam=1.0,
    )
    np.testing.assert_allclose(advantages[:, 0], [2.5, 1.3, -93.0], atol=1e-5)


def test_gae_bootstraps_unfinished_rollout():
    advantages = compute_gae(
        jnp.array([[1.0]]), jnp.array([[2.0]]),
        jnp.array([[False]]), jnp.array([10.0]), gamma=0.9,
    )
    np.testing.assert_allclose(advantages, [[8.0]])


def test_training_updates_parameters_and_saves_model(tmp_path):
    network = PolicyValueNetwork(jax.random.PRNGKey(0))
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
    obs = jnp.ones((9, 4, 4))
    mask = jnp.ones((4, 4, 4), dtype=bool)
    action, value, logprob, _ = network(obs, mask, jax.random.PRNGKey(1))
    batch = tuple(x[None, None] for x in (
        obs, mask, action, logprob, jnp.array(1.0), value + 1.0,
    ))
    updated, _, loss = train_step(network, opt_state, batch, optimizer)
    assert bool(jnp.isfinite(loss))
    before = jax.tree.leaves(eqx.filter(network, eqx.is_array))
    after = jax.tree.leaves(eqx.filter(updated, eqx.is_array))
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))
    path = tmp_path / "model.eqx"
    eqx.tree_serialise_leaves(path, updated)
    restored = eqx.tree_deserialise_leaves(path, network)
    actual = restored(obs, mask, None, action)
    expected = updated(obs, mask, None, action)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)

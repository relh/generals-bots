"""Frozen training opponents honor the same action mode as their serving bundle."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from integrations.spatial_policy_bundle import structured_action_probabilities
from integrations.spatial_selfplay import frozen_action_indices


def test_frozen_sampled_opponent_matches_serving_distribution():
    count = 20_000
    outputs = np.zeros((3530,), np.float32)
    outputs[0], outputs[1764], outputs[3528] = .15, 0, .1
    legal = np.zeros((3529,), bool)
    legal[[0, 1764, 3528]] = True
    expected = structured_action_probabilities(outputs, legal, .05, .15)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15)
    keys = jax.random.split(jax.random.PRNGKey(123), count)
    choose = jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k
    ))
    sampled = np.asarray(choose(keys))
    assert np.isin(sampled, [0, 1764, 3528]).all()
    np.testing.assert_array_equal(sampled, np.asarray(choose(keys)))
    for action in (0, 1764, 3528):
        assert abs(np.mean(sampled == action) - expected[action]) < .02


def test_frozen_argmax_opponent_keeps_legacy_selection():
    outputs = np.zeros((2, 3530), np.float32)
    outputs[:, 0], outputs[:, 1764], outputs[:, 3528] = .15, 0, .1
    legal = np.zeros((2, 3529), bool)
    legal[:, [1764, 3528]] = True
    chosen = frozen_action_indices(SimpleNamespace(action_mode="argmax"),
                                   jnp.asarray(outputs), jnp.asarray(legal),
                                   jax.random.split(jax.random.PRNGKey(7), 2))
    np.testing.assert_array_equal(np.asarray(chosen), [3528, 3528])


def test_frozen_neutral_route_bonus_matches_serving_distribution():
    count = 10_000
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs = np.zeros(3530, np.float32)
    legal = np.zeros(3529, bool)
    legal[[up, right]] = True
    public = np.zeros(16 * 441, np.float32)
    public[6 * 441 + source + 1] = 1
    expected = structured_action_probabilities(outputs, legal, .05, .15,
                                               observations=public, neutral_route_bias=3.0)
    policy = SimpleNamespace(action_mode="structured_sample", move_temperature=.05,
                             split_temperature=.15, neutral_route_bias=3.0)
    keys = jax.random.split(jax.random.PRNGKey(109), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(outputs, (count, 3530)),
        jnp.broadcast_to(legal, (count, 3529)), k,
        jnp.broadcast_to(public, (count, public.size)),
    ))(keys))
    assert abs(np.mean(sampled == up) - expected[up]) < .02
    with pytest.raises(ValueError, match="public observations"):
        frozen_action_indices(policy, jnp.asarray(outputs[None]), jnp.asarray(legal[None]),
                              keys[:1])

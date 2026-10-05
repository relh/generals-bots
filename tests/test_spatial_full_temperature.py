"""Full exploration temperature must preserve masked PPO/serving semantics."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from integrations.spatial_action_sampling import (
    acting_logits,
    public_neutral_route_bonus,
    raw_cotangents,
    scale_action_logits,
)
from integrations.spatial_frozen_sampling import frozen_action_indices
from integrations.spatial_policy_bundle import structured_action_probabilities


@pytest.mark.parametrize('temperature', [0, -1, np.inf, np.nan, True, '10'])
def test_reject_invalid_full_temperature(temperature):
    with pytest.raises(ValueError, match='Full action temperature'):
        scale_action_logits(np.zeros(3529), temperature)


def test_complete_transform_vjp_preserves_value_and_scales_action_gradients():
    rng = np.random.default_rng(184)
    raw = jnp.asarray(rng.normal(0, .1, (2, 1, 3530)), jnp.float32)
    actions = jnp.asarray(rng.normal(0, .01, (2, 1, 3529)), jnp.float32)
    values = jnp.asarray([[-.2], [.7]], jnp.float32)
    temperatures = jnp.asarray([.1, .05])[:, None, None]
    # Fixed public priors enter BEFORE scaling; derivatives must match end to end.
    priors = jnp.asarray(rng.normal(0, 4, (2, 1, 3529)), jnp.float32)
    def transform(predictions):
        result = acting_logits(predictions, temperatures, .15, jnp, route_half_weight=.25)
        return jnp.concatenate((scale_action_logits(result[..., :3529] + priors, 10),
                                result[..., 3529:]), axis=-1)
    result, backward = jax.vjp(transform, raw)
    actual = backward(jnp.concatenate((actions, values[..., None]), axis=-1))[0]
    expected = raw_cotangents(raw, actions / 10, values, temperatures, .15, jnp,
                              route_half_weight=.25)
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-5)
    np.testing.assert_array_equal(result[..., -1], raw[..., -1])
    np.testing.assert_array_equal(actual[..., -1], values)


def test_full_temperature_matches_serving_and_frozen_sampler_with_priors():
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    raw = np.zeros(3530, np.float32)
    raw[[up, right, up + 1764, right + 1764, 3528]] = [.1, -.1, -.2, -.4, 0]
    mask = np.zeros(3529, bool)
    mask[[up, right, up + 1764, right + 1764, 3528]] = True
    public = np.zeros(16 * 441, np.float32)
    public[6 * 441 + source + 1] = 1
    expected_logits = (acting_logits(raw, .05, .15, np)[:3529]
                       + public_neutral_route_bonus(public, 6, np)) / 10
    logits = np.where(mask, expected_logits, -np.inf)
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    portable = structured_action_probabilities(raw, mask, .05, .15,
        observations=public, neutral_route_bias=6, full_action_temperature=10, log_gap_scale=0.)
    np.testing.assert_allclose(portable, expected, atol=1e-8, rtol=1e-6)
    assert np.all(portable[~mask] == 0)
    policy = SimpleNamespace(action_mode='structured_sample', move_temperature=.05,
        early_route_temperature=None, early_route_turns=None, route_half_weight=0.,
        weak_owned_route_penalty=0., doomed_attack_route_penalty=0.,
        split_temperature=.15, neutral_route_bias=6, full_action_temperature=10, log_gap_scale=0.)
    count = 8192
    keys = jax.random.split(jax.random.PRNGKey(31), count)
    sampled = np.asarray(jax.jit(lambda k: frozen_action_indices(policy,
        jnp.broadcast_to(raw, (count, 3530)), jnp.broadcast_to(mask, (count, 3529)), k,
        jnp.broadcast_to(public, (count, len(public)))))(keys))
    assert mask[sampled].all()
    for action in np.flatnonzero(mask):
        assert abs(np.mean(sampled == action) - expected[action]) < .025
    # The old move-only temperature knob is not the new full-logit transform.
    move_only = structured_action_probabilities(raw, mask, .5, .15,
        observations=public, neutral_route_bias=6)
    assert np.max(np.abs(move_only - portable)) > .1


def test_default_temperature_preserves_original_logits_exactly():
    raw = np.arange(3529, dtype=np.float32)
    assert scale_action_logits(raw, 1) is raw

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.neural_codec import training_observation
from integrations.spatial_action_sampling import acting_logits, raw_cotangents
from integrations.spatial_exploration import (
    log_gap_cotangents,
    log_gap_logits,
    public_action_mask,
)
from integrations.spatial_frozen_sampling import frozen_action_indices
from integrations.spatial_policy_bundle import structured_action_probabilities


@pytest.mark.parametrize('height,width', [(18, 21), (21, 18), (19, 20), (21, 21)])
def test_public_mask_matches_codec_on_visible_edges_and_fog(height, width):
    rng = np.random.default_rng(height * 100 + width)
    owned = rng.random((height, width)) < .08
    owned[0, 0] = owned[-1, -1] = owned[0, -1] = owned[-1, 0] = True
    padded = np.pad(owned, 1)
    visible = np.zeros_like(owned)
    for dr in range(3):
        for dc in range(3):
            visible |= padded[dr:dr+height, dc:dc+width]
    kinds = np.where(visible, rng.choice([1, 2, 3], (height, width)),
                     rng.choice([0, 5], (height, width)))
    kinds[owned] = 1
    owners = np.where(owned, 1, np.where(visible & (kinds == 1), 2, 0))
    armies = np.where(owned, rng.integers(1, 5, (height, width)), 0)
    armies[0, 0] = 1  # Must not create moves for a singleton army.
    armies[-1, -1] = 2  # Both full/half routes remain legal at the boundary.
    message = dict(height=height, width=width, type_grid=kinds.tolist(),
                   owner_grid=owners.tolist(), army_grid=armies.tolist(),
                   my_land=int(owned.sum()), my_army=int(armies.sum()),
                   opp_land=int((owners == 2).sum()), opp_army=0, turn=1999)
    observation = training_observation(message)
    values, expected = encode_coworld_directional_observation(observation)
    np.testing.assert_array_equal(public_action_mask(np.asarray(values), np), expected)
    np.testing.assert_array_equal(jax.jit(lambda x: public_action_mask(x, jnp))(values), expected)


@pytest.mark.parametrize('scale', [2., 4., 8.])
def test_exact_vjp_with_tied_maxima_and_masked_larger_logits(scale):
    raw = jnp.array([[3., 3., -70., 10000.], [-20., 2., -1., 999.]], jnp.float32)
    legal = jnp.array([[True, True, True, False], [False, True, False, False]])
    cotangent = jnp.array([[.2, -.4, .7, 123.], [4., 5., 6., 7.]], jnp.float32)
    _, backward = jax.vjp(lambda x: log_gap_logits(x, legal, scale, jnp), raw)
    expected = backward(cotangent)[0]
    actual = log_gap_cotangents(raw, legal, cotangent, scale, jnp)
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(actual[~legal], 0)
    np.testing.assert_allclose(actual.sum(axis=-1), 0, atol=1e-7)
    np.testing.assert_array_equal(actual[1], 0)  # Only one possible action.


def test_shift_invariance_rankings_and_rare_action_probability():
    logits = np.array([0., -10., -70., 1e6])
    legal = np.array([True, True, True, False])
    transformed = log_gap_logits(logits, legal, 4., np)
    np.testing.assert_allclose(transformed, log_gap_logits(logits + 100., legal, 4., np))
    np.testing.assert_array_equal(np.argsort(transformed[legal]), np.argsort(logits[legal]))
    probability = np.exp(transformed[legal])
    probability /= probability.sum()
    assert probability[2] > 1e-6
    assert np.isfinite(transformed).all()


def test_zero_scale_preserves_legacy_logits_and_cotangents():
    raw, mask, gradients = np.array([2., -7.]), np.array([True, False]), np.array([.2, .8])
    assert log_gap_logits(raw, mask, 0, np) is raw
    assert log_gap_cotangents(raw, mask, gradients, 0, np) is gradients


@pytest.mark.parametrize('scale', [-1, True, np.inf, np.nan, '4'])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match='Log gap scale'):
        log_gap_logits(np.zeros(3), np.ones(3, bool), scale, np)


def test_unrecognized_observation_layout_rejected():
    with pytest.raises(ValueError, match='sixteen'):
        public_action_mask(np.zeros(11 * 441), np)


def test_complete_ppo_chain_rule_preserves_value_gradients():
    rng = np.random.default_rng(579)
    raw = jnp.asarray(rng.normal(0, .1, (2, 1, 3530)), jnp.float32)
    public = np.zeros((2, 1, 16, 441), np.float32)
    public[..., 4, 220] = 1
    public[..., 0, 220] = np.log1p(25) / 8
    legal = public_action_mask(jnp.asarray(public.reshape(2, 1, -1)), jnp)
    priors = jnp.asarray(rng.normal(0, 3, (2, 1, 3529)), jnp.float32)
    actions = jnp.asarray(rng.normal(0, .02, (2, 1, 3529)), jnp.float32)
    values = jnp.asarray([[.7], [-.3]])

    def transform(predictions):
        acting = acting_logits(predictions, .05, .15, jnp, route_half_weight=.25)
        logits = (acting[..., :3529] + priors) / 2
        return jnp.concatenate((log_gap_logits(logits, legal, 4, jnp), acting[..., -1:]), axis=-1)

    result, backward = jax.vjp(transform, raw)
    actual = backward(jnp.concatenate((actions, values[..., None]), axis=-1))[0]
    before = (acting_logits(raw, .05, .15, jnp, route_half_weight=.25)[..., :3529] + priors) / 2
    expected_actions = log_gap_cotangents(before, legal, actions, 4, jnp) / 2
    expected = raw_cotangents(raw, expected_actions, values, .05, .15, jnp, route_half_weight=.25)
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-5)
    np.testing.assert_array_equal(result[..., -1], raw[..., -1])
    np.testing.assert_array_equal(actual[..., -1], values)


def test_frozen_sampling_frequencies_match_portable_probabilities():
    public = np.zeros((16, 441), np.float32)
    public[4, 220] = 1
    public[0, 220] = np.log1p(12) / 8
    public = public.reshape(-1)
    legal = public_action_mask(public, np)
    raw = np.random.default_rng(957).normal(0, .15, 3530).astype(np.float32)
    expected = structured_action_probabilities(raw, legal, .05, .15,
                                              observations=public, log_gap_scale=4)
    policy = SimpleNamespace(action_mode='structured_sample', move_temperature=.05,
        early_route_temperature=None, early_route_turns=None, route_half_weight=0.,
        weak_owned_route_penalty=0., doomed_attack_route_penalty=0.,
                             split_temperature=.15, log_gap_scale=4, full_action_temperature=1., neutral_route_bias=0.)
    count = 8192
    keys = jax.random.split(jax.random.PRNGKey(7), count)
    samples = np.asarray(jax.jit(lambda k: frozen_action_indices(
        policy, jnp.broadcast_to(raw, (count, 3530)), jnp.broadcast_to(legal, (count, 3529)),
        k, jnp.broadcast_to(public, (count, 7056))))(keys))
    assert legal[samples].all()
    for action in np.flatnonzero(legal):
        assert abs(np.mean(samples == action) - expected[action]) < .02
    corrupted = legal.copy()
    corrupted[3528] = False
    with pytest.raises(ValueError, match='Public exploration mask differs'):
        structured_action_probabilities(raw, corrupted, .05, .15, observations=public, log_gap_scale=4)

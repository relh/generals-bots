"""The 4→8 ablation changes one existing public route prior, without hard masks."""
from types import SimpleNamespace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.distill_defense import serving_logits
from integrations.spatial_action_sampling import public_doomed_attack_route_penalty
from integrations.spatial_policy_bundle import structured_action_probabilities


def view():
    planes = np.zeros((16, 441), np.float32)
    source = 10 * 21 + 10
    planes[0, source] = np.log1p(11) / 8
    planes[4, source] = 1
    for target, army in ((source - 21, 10), (source + 21, 9)):
        planes[0, target] = np.log1p(army) / 8
        planes[5, target] = 1
    planes[4, source - 1] = 1
    return planes.reshape(-1), source


def policy(strength):
    return SimpleNamespace(move_temperature=1., split_temperature=.15,
        early_route_temperature=None, neutral_route_bias=0.,
        weak_owned_route_penalty=0., doomed_attack_route_penalty=strength,
        full_action_temperature=1., route_half_weight=0., log_gap_scale=0., capital_safety=False)


def test_public_tie_penalty_numpy_jax_parity_and_legal_chip_damage():
    values, source = view()
    route = source  # Up: full transfers10, tie against10 does not capture.
    safe_route = 441 + source  # Down: full transfers10 captures9.
    mask = np.zeros(3529, bool)
    mask[[route, safe_route, route + 1764, safe_route + 1764, 3528]] = True
    raw = np.zeros(3530, np.float32)
    prior = public_doomed_attack_route_penalty(values, 8., np)
    np.testing.assert_array_equal(prior, public_doomed_attack_route_penalty(jnp.asarray(values), 8., jnp))
    assert prior[route] == prior[route + 1764] == -8
    assert prior[safe_route] == prior[safe_route + 1764] == prior[3528] == 0
    distributions = []
    for strength in (4., 8.):
        actual = structured_action_probabilities(raw, mask, 1., .15, observations=values,
                                                 doomed_attack_route_penalty=strength)
        logits = serving_logits(policy(strength), jnp.asarray(raw)[None],
                                jnp.asarray(values)[None], jnp.asarray(mask)[None])[0]
        np.testing.assert_allclose(actual, jax.nn.softmax(logits), atol=1e-7)
        assert np.all(actual[~mask] == 0) and actual[route] > 0 and actual[route + 1764] > 0
        # Equal full/half penalty preserves the conditional split distribution.
        np.testing.assert_allclose(actual[route] / actual[route + 1764], 1.)
        distributions.append(actual)
    before, after = distributions
    np.testing.assert_allclose((after[route] / after[safe_route]) /
                               (before[route] / before[safe_route]), np.exp(-4), rtol=1e-6)


def test_no_visible_doomed_enemy_no_change_and_only_one_config_delta():
    values, _ = view()
    planes = values.reshape(16, 441).copy()
    planes[5] = 0  # Hidden/non-enemy terrain cannot activate the prior.
    values = planes.reshape(-1)
    np.testing.assert_array_equal(public_doomed_attack_route_penalty(values, 8., np), np.zeros(3529))
    raw = np.linspace(-1, 1, 3530, dtype=np.float32)
    mask = np.ones(3529, bool)
    a = structured_action_probabilities(raw, mask, 1., .15, observations=values, doomed_attack_route_penalty=4.)
    b = structured_action_probabilities(raw, mask, 1., .15, observations=values, doomed_attack_route_penalty=8.)
    np.testing.assert_array_equal(a, b)
    experiment = json.loads((Path(__file__).parents[1] / 'experiments/doomed_attack_penalty.json').read_text())
    assert experiment['changed_sampler_field'] == 'doomed_attack_route_penalty'
    assert experiment['baseline'] == 4 and experiment['candidate'] == 8
    assert experiment['capital_safety'] is False and experiment['policy_weights_unchanged']

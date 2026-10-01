import numpy as np
import pytest

from integrations.spatial_policy_bundle import structured_action_probabilities
from integrations.spatial_action_sampling import (acting_logits, public_owned_split_bias,
                                                  public_weak_owned_route_penalty)


def test_structured_serving_probabilities_follow_route_and_split_logits():
    outputs = np.full(3530, -100.0, np.float32)
    outputs[0], outputs[1764] = 0.0, -0.3
    outputs[1], outputs[1765] = -0.2, -0.5
    legal = np.zeros(3529, bool)
    legal[[0, 1, 1764, 1765]] = True

    probabilities = structured_action_probabilities(outputs, legal, .05, .15)

    assert probabilities.shape == (3529,)
    assert probabilities[~legal].sum() == 0
    assert probabilities.sum() == pytest.approx(1)
    assert probabilities[0] / probabilities[1764] == pytest.approx(np.exp(2), rel=1e-5)
    assert (probabilities[0] + probabilities[1764]) / (
        probabilities[1] + probabilities[1765]
    ) == pytest.approx(np.exp(4), rel=1e-5)

    legal[1764] = False
    masked = structured_action_probabilities(outputs, legal, .05, .15)
    assert masked[1764] == 0
    assert masked.sum() == pytest.approx(1)


def test_structured_serving_rejects_invalid_distribution():
    outputs = np.zeros(3530, np.float32)
    legal = np.zeros(3529, bool)
    with pytest.raises(ValueError, match="Invalid spatial logits"):
        structured_action_probabilities(outputs, legal, .05, .15)
    legal[0] = True
    with pytest.raises(ValueError, match="action temperatures"):
        structured_action_probabilities(outputs, legal, 0, .15)


def test_public_neutral_route_bonus_changes_route_without_changing_split():
    outputs = np.full(3530, -100.0, np.float32)
    # Two legal moves from the same source: up reaches neutral, right is fog.
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs[[up, right]] = 0
    outputs[[1764 + up, 1764 + right]] = -.3
    legal = np.zeros(3529, bool)
    legal[[up, right, 1764 + up, 1764 + right]] = True
    public = np.zeros(16 * 441, np.float32)
    public[6 * 441 + source + 1] = 1
    baseline = structured_action_probabilities(outputs, legal, .05, .15)
    biased = structured_action_probabilities(outputs, legal, .05, .15,
                                             observations=public, neutral_route_bias=3.0)
    assert (biased[up] + biased[1764 + up]) / (biased[right] + biased[1764 + right]) == pytest.approx(np.exp(3))
    assert biased[up] / biased[1764 + up] == pytest.approx(baseline[up] / baseline[1764 + up])
    with pytest.raises(ValueError, match="public observations"):
        structured_action_probabilities(outputs, legal, .05, .15, neutral_route_bias=3.0)


def test_owned_split_bias_preserves_route_mass_and_uses_public_army():
    source = 10 * 21 + 10
    up, right = source, 3 * 441 + source
    outputs = np.zeros(3530, np.float32)
    outputs[[1764 + up, 1764 + right]] = -.3
    public = np.zeros(16 * 441, np.float32)
    public[4 * 441 + source - 21] = 1
    public[source] = np.log1p(5) / 8
    split_bias = public_owned_split_bias(public, 4.0, np)
    assert split_bias[up] == 4.0
    assert split_bias[right] == 0.0
    before = acting_logits(outputs, .05, .15, np)
    after = acting_logits(outputs, .05, .15, np, split_bias)
    assert np.logaddexp(after[up], after[1764 + up]) == pytest.approx(
        np.logaddexp(before[up], before[1764 + up]), abs=1e-6)
    assert (after[1764 + up] - after[up]) == pytest.approx(
        before[1764 + up] - before[up] + 4.0, abs=1e-6)
    public[source] = np.log1p(4) / 8
    assert public_owned_split_bias(public, 4.0, np)[up] == 0


def test_weak_owned_route_penalty_requires_early_land_and_small_stack():
    source = 10 * 21 + 10
    up = source
    public = np.zeros(16 * 441, np.float32)
    public[4 * 441 + source] = 1
    public[4 * 441 + source - 21] = 1
    public[source] = np.log1p(4) / 8
    penalty = public_weak_owned_route_penalty(public, 4.0, np)
    assert penalty[up] == penalty[1764 + up] == -4.0
    assert penalty[3528] == 0
    public[source] = np.log1p(5) / 8
    assert public_weak_owned_route_penalty(public, 4.0, np)[up] == 0
    public[source] = np.log1p(4) / 8
    public[4 * 441:4 * 441 + 15] = 1
    assert public_weak_owned_route_penalty(public, 4.0, np)[up] == 0

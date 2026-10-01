import numpy as np
import pytest

from integrations.spatial_policy_bundle import structured_action_probabilities


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

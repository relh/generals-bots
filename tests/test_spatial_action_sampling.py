import numpy as np

from integrations.spatial_action_sampling import MOVE_COUNT, acting_logits, raw_cotangents


def test_split_sampling_keeps_route_marginals_and_exact_vjp():
    rng = np.random.default_rng(91)
    raw = rng.normal(0, .2, (2, 3530))
    raw[:, MOVE_COUNT:2 * MOVE_COUNT] = raw[:, :MOVE_COUNT] - .4
    gradients = rng.normal(0, .01, (2, 3529))
    values = rng.normal(0, .01, (2,))
    move_temperature, split_temperature = .05, .25
    acting = acting_logits(raw, move_temperature, split_temperature, np)
    np.testing.assert_allclose(
        np.logaddexp(acting[:, :MOVE_COUNT], acting[:, MOVE_COUNT:2 * MOVE_COUNT]),
        raw[:, :MOVE_COUNT] / move_temperature, rtol=1e-14, atol=1e-14,
    )
    np.testing.assert_allclose(acting[:, 3528], raw[:, 3528] / move_temperature)
    np.testing.assert_allclose(acting[:, 3529], raw[:, 3529])
    half_probability = np.exp(acting[:, MOVE_COUNT:2 * MOVE_COUNT] - raw[:, :MOVE_COUNT] / move_temperature)
    np.testing.assert_allclose(half_probability, 1 / (1 + np.exp(.4 / split_temperature)))

    analytic = raw_cotangents(raw, gradients, values, move_temperature, split_temperature, np)
    def objective(predictions):
        result = acting_logits(predictions, move_temperature, split_temperature, np)
        return float(np.sum(result[:, :3529] * gradients) + np.sum(result[:, 3529] * values))

    for row, index in ((0, 0), (0, MOVE_COUNT), (0, 442), (1, MOVE_COUNT + 442),
                       (1, 3528), (1, 3529)):
        left, right = raw.copy(), raw.copy()
        left[row, index] -= 1e-5
        right[row, index] += 1e-5
        numeric = (objective(right) - objective(left)) / 2e-5
        np.testing.assert_allclose(analytic[row, index], numeric, rtol=2e-5, atol=2e-7)

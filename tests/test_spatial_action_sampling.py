import numpy as np

from integrations.spatial_action_sampling import (MOVE_COUNT, acting_logits, public_early_route_temperature,
                                                  raw_cotangents)


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


def test_public_turn_schedule_preserves_per_row_ppo_cotangents():
    observations = np.zeros((2, 1, 16 * 441), np.float32)
    observations[0, 0, 11 * 441] = 99 / 2000
    observations[1, 0, 11 * 441] = 100 / 2000
    temperatures = public_early_route_temperature(observations, .05, .1, 100, np)
    np.testing.assert_array_equal(temperatures[:, 0, 0], [.1, .05])

    rng = np.random.default_rng(7)
    predictions = rng.normal(0, .1, (2, 1, 3530))
    logits = rng.normal(0, .01, (2, 1, 3529))
    values = rng.normal(0, .01, (2, 1))
    analytic = raw_cotangents(predictions, logits, values, temperatures, .15, np)
    def objective(raw):
        output = acting_logits(raw, temperatures, .15, np)
        return float(np.sum(output[..., :3529] * logits) + np.sum(output[..., 3529] * values))
    for row, action in ((0, 0), (0, 1764), (1, 441), (1, 3528)):
        left, right = predictions.copy(), predictions.copy()
        left[row, 0, action] -= 1e-5
        right[row, 0, action] += 1e-5
        np.testing.assert_allclose(analytic[row, 0, action], (objective(right) - objective(left)) / 2e-5,
                                   rtol=2e-5, atol=2e-7)

import numpy as np
import jax.numpy as jnp

from integrations.concentration_potential import concentration, potential, shaping


def state(stacks):
    armies = np.asarray([stacks, [0] * len(stacks)], np.int32)
    own = np.zeros((2, 2, len(stacks)), bool)
    own[0, 0] = armies[0] > 0
    own[1, 1, 0] = True
    armies[1, 0] = 40
    return armies, own


def test_merge_credit_transport_neutrality_and_numpy_jax_parity():
    old = state([20, 20])
    merged = state([1, 39])
    transported = state([39, 1])
    a, b = [potential(*s, 100, np) for s in (old, merged)]
    np.testing.assert_allclose(concentration(*old, np), concentration(*old, jnp))
    np.testing.assert_allclose(b, potential(*merged, 100, jnp))
    assert 0 <= concentration(*old, np).min() <= concentration(*old, np).max() <= 1
    assert shaping(a, b, False)[0] > .0225
    np.testing.assert_array_equal(b, potential(*transported, 100, np))
    # Floating arithmetic prevents integer overflow at large stacks.
    assert np.isfinite(concentration(*state([100000, 100000]), np)).all()


def test_discounted_cycle_and_terminal_cannot_farm_shaping():
    gamma = .999
    before = potential(*state([20, 20]), 100, np)
    after = potential(*state([1, 39]), 100, np)
    cycle = shaping(before, after, False) + gamma * shaping(after, before, False)
    np.testing.assert_allclose(cycle, .05 * (gamma**2 - 1) * before, atol=1e-8)
    terminal = shaping(after, np.ones(2) * 100, True)
    np.testing.assert_allclose(terminal, -.05 * after)
    trajectory = shaping(before, after, False) + gamma * terminal
    np.testing.assert_allclose(trajectory, -.05 * before, atol=1e-8)


def test_time_boundary_is_a_potential_transition_not_free_merge_bonus():
    before = potential(*state([20, 20]), 99, np)
    after = potential(*state([1, 39]), 100, np)
    np.testing.assert_array_equal(before, np.zeros(2))
    assert np.any(shaping(before, after, False) != 0)  # Activation creates boundary credit.
    terminal = shaping(after, after, True)
    np.testing.assert_allclose(shaping(before, after, False) + .999 * terminal, 0, atol=1e-8)

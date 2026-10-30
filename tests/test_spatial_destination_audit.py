import jax.numpy as jnp
import numpy as np

from integrations.spatial_destination_audit import destination_categories


def test_flat_full_half_and_pass_destinations_use_pre_action_ownership():
    neutral = np.ones((5, 21, 21), bool)
    ownership = np.zeros((5, 2, 21, 21), bool)
    neutral[1, 9, 10] = False
    ownership[1, 1, 9, 10] = True
    neutral[2, 10, 11] = False
    ownership[2, 1, 10, 11] = True
    source = 10 * 21 + 10
    actions = jnp.asarray((source, 1764 + source, 3 * 441 + source, 3528, source))
    result = destination_categories(
        jnp.asarray(neutral), jnp.asarray(ownership),
        jnp.asarray((0, 1, 0, 1, 0)), actions,
        jnp.asarray((True, True, True, True, False)),
    )
    assert result.tolist() == [1, 2, 3, 4, 0]

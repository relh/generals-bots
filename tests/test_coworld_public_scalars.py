"""Public wire and training views share the fixed Classic economic features."""
import numpy as np
import pytest

from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.neural_codec import encode_wire_observation, training_observation


@pytest.mark.parametrize("height,width", [(18, 21), (21, 18), (19, 20), (21, 21)])
def test_public_scalar_wire_training_contract(height, width):
    kinds = np.ones((height, width), np.int32)
    owners, armies = np.zeros_like(kinds), np.zeros_like(kinds)
    kinds[2, 2], owners[2, 2], armies[2, 2] = 4, 1, 20
    owners[2, 3], armies[2, 3] = 1, 10
    kinds[-3, -3], owners[-3, -3], armies[-3, -3] = 4, 2, 20
    message = dict(height=height, width=width, type_grid=kinds.tolist(),
                   owner_grid=owners.tolist(), army_grid=armies.tolist(),
                   my_land=2, my_army=30, opp_land=50, opp_army=5000, turn=1000)
    values, mask = encode_wire_observation(message)
    assert values.shape == (7056,) and mask.shape == (3529,)
    expected = np.array([.5, 2/441, 50/441, np.log1p(30)/8, np.log1p(5000)/8], np.float32)
    np.testing.assert_allclose(values.reshape(16, 441)[11:], np.broadcast_to(expected[:, None], (5, 441)), rtol=1e-6)
    direct, direct_mask = encode_coworld_directional_observation(training_observation(message))
    np.testing.assert_allclose(values, direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(mask, direct_mask)

import io

import jax.numpy as jnp
import numpy as np

from competition.agents.sentinel_python.main import read_observation
from competition.protocol import encode_observation
from generals.core import game


def test_wire_observation_preserves_policy_inputs():
    grid = jnp.array([[1, -2, 0, 0], [0, 0, 20, 0], [0, 0, -2, 2]], dtype=jnp.int32)
    for player in (0, 1):
        original = game.get_observation(game.create_initial_state(grid), player)
        actual = read_observation(io.StringIO(encode_observation(original)), 3, 4)
        for field in original._fields:
            np.testing.assert_array_equal(getattr(actual, field), getattr(original, field))


def test_end_of_stream_is_normal_shutdown():
    assert read_observation(io.StringIO(''), 3, 4) is None

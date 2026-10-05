import numpy as np
import pytest

from integrations.spatial_context_geometry import context_offsets, kernel_offsets
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def test_neighborhoods_and_unsupported_geometry():
    assert context_offsets(1.01) == ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
    assert len(context_offsets(2.01)) == 13
    assert (2, 0) in context_offsets(2.01) and (2, 1) not in context_offsets(2.01)
    for radius in [True, 1, 2, 2.5, float("nan")]:
        with pytest.raises(ValueError):
            context_offsets(radius)
    with pytest.raises(ValueError):
        kernel_offsets(7)


def toy_policy(size):
    policy = object.__new__(SpatialPlayerPolicy)
    policy.channels = 16
    policy.features = 1
    policy.observation_size = 7056
    policy.prior_count = 0
    policy.weights = dict(
        input_kernel=np.concatenate((np.ones((1, 1), np.float32), np.zeros((15, 1), np.float32))),
        local_weight=np.ones(1, np.float32),
        local_bias=np.zeros(1, np.float32),
        context_kernel=np.zeros((size, size, 1, 1), np.float32),
        context_weight=np.ones(1, np.float32),
        context_bias=np.zeros(1, np.float32),
        global_kernel=np.zeros((441, 1), np.float32),
        global_weight=np.ones(1, np.float32),
        global_bias=np.zeros(1, np.float32),
        readout_kernel=np.zeros((1, 3530), np.float32),
        action_kernel=np.ones((1, 8), np.float32),
        output_weight=np.ones(3530, np.float32),
        output_bias=np.zeros(3530, np.float32),
    )
    return policy


def test_new_neighbor_can_influence_action_without_board_wraparound():
    policy = toy_policy(5)
    # The output site receives a signal from two columns to its left.
    policy.weights["context_kernel"][2, 0, 0, 0] = 1
    obs = np.zeros((1, 7056), np.float32)
    obs[0, 0] = 2
    output = policy.forward(obs)[0, :441].reshape(21, 21)
    assert output[0, 2] > 0
    assert np.count_nonzero(output) == 1
    obs[:] = 0
    obs[0, 20] = 2
    assert np.count_nonzero(policy.forward(obs)) == 0

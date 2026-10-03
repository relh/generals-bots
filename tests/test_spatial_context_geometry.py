
import numpy as np
import pytest

from integrations.spatial_context_geometry import context_offsets, kernel_offsets
from integrations.spatial_context_transfer import LEARNER_HEADER, extend_flat, extend_learner
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
    policy.channels = policy.features = 1
    policy.observation_size = 441
    policy.public_scalar_ablation = False
    policy.prior_count = 0
    policy.weights = dict(input_kernel=np.ones((1, 1), np.float32),
                          local_weight=np.ones(1, np.float32), local_bias=np.zeros(1, np.float32),
                          context_kernel=np.zeros((size, size, 1, 1), np.float32),
                          context_weight=np.ones(1, np.float32), context_bias=np.zeros(1, np.float32),
                          global_kernel=np.zeros((441, 1), np.float32), global_weight=np.ones(1, np.float32),
                          global_bias=np.zeros(1, np.float32), readout_kernel=np.zeros((1, 3530), np.float32),
                          action_kernel=np.ones((1, 8), np.float32), output_weight=np.ones(3530, np.float32),
                          output_bias=np.zeros(3530, np.float32))
    return policy


def test_new_neighbor_can_influence_action_without_board_wraparound():
    policy = toy_policy(5)
    # The output site receives a signal from two columns to its left.
    policy.weights["context_kernel"][2, 0, 0, 0] = 1
    obs = np.zeros((1, 441), np.float32)
    obs[0, 0] = 2
    output = policy.forward(obs)[0, :441].reshape(21, 21)
    assert output[0, 2] > 0
    assert np.count_nonzero(output) == 1
    obs[:] = 0
    obs[0, 20] = 2
    assert np.count_nonzero(policy.forward(obs)) == 0


def test_zero_extension_preserves_serving_on_nontrivial_observations():
    old, new = toy_policy(3), toy_policy(5)
    for y, x in kernel_offsets(3):
        old.weights["context_kernel"][y, x, 0, 0] = (y * 3 + x + 1) / 10
    new.weights["context_kernel"][1:4, 1:4] = old.weights["context_kernel"]
    views = np.random.default_rng(424).normal(size=(3, 441)).astype(np.float32)
    np.testing.assert_array_equal(old.forward(views), new.forward(views))


def test_same_mapping_preserves_weights_and_momentum_with_zero_new_parameters():
    mapping = np.array([2, -1, 0, -1, 1], np.int32)
    for source in [np.array([1., 2., 3.], np.float32), np.array([-.1, .2, -.3], np.float32)]:
        result = extend_flat(source, mapping)
        np.testing.assert_array_equal(result[[2, 4, 0]], source)
        np.testing.assert_array_equal(result[[1, 3]], 0)
    with pytest.raises(ValueError, match="exceeds"):
        extend_flat(np.ones(2, np.float32), mapping)
    with pytest.raises(ValueError, match="finite"):
        extend_flat(np.array([np.nan], np.float32), np.array([0]))


def test_learner_extension_preserves_clocks_and_rejects_bad_snapshots():
    mapping = np.array([2, -1, 0, -1, 1], np.int32)
    header = LEARNER_HEADER.pack(b"METTAL01", 7, 56, 3, .0002)
    data = header + np.array([-.1, .2, -.3], dtype="<f4").tobytes()

    def transfer(value):
        return extend_learner(value, mapping, source_count=3, expected_steps=56, batch_steps=8)

    result = transfer(data)
    assert LEARNER_HEADER.unpack_from(result)[:4] == (b"METTAL01", 7, 56, 5)
    assert result[32:36] == data[32:36]  # Preserve exact float32 LR bytes.
    np.testing.assert_array_equal(np.frombuffer(result, "<f4", offset=36),
                                  np.array([-.3, 0, -.1, 0, .2], np.float32))
    for bad in [data[:20], data[:-1], data + b"x", b"INVALID!" + data[8:],
                LEARNER_HEADER.pack(b"METTAL01", 6, 56, 3, .0002) + data[36:],
                LEARNER_HEADER.pack(b"METTAL01", 7, 56, 3, float("nan")) + data[36:],
                header + np.array([1, np.inf, 2], dtype="<f4").tobytes()]:
        with pytest.raises(ValueError):
            transfer(bad)

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from integrations.sparse_teacher import SparseActionTeacher


def test_sparse_teacher_uses_only_labeled_legal_actions():
    objective = SparseActionTeacher(action_sizes=[3, 2])
    policy = jnp.asarray([[[0.0, 1.0, 2.0, 0.0, 2.0], [2.0, 0.0, 1.0, 1.0, 0.0]]])
    replay = SimpleNamespace(
        metadata=np.asarray([[[2.0, 1.0], [0.0, -1.0]]], np.float32),
        action_masks=np.asarray([[[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]], np.float32),
    )
    loss, gradient = jax.value_and_grad(lambda x: objective(x, jnp.empty((1, 2, 0)), replay, seed=(0, 0)))(policy)
    first = -jax.nn.log_softmax(jnp.asarray([0.0, -1e9, 2.0]))[2]
    second = -jax.nn.log_softmax(jnp.asarray([2.0, 0.0, 1.0]))[0]
    split = -jax.nn.log_softmax(jnp.asarray([0.0, 2.0]))[1]
    np.testing.assert_allclose(loss, (first + second) / 2 + split, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(gradient)[0, 1, 3:], 0)
    np.testing.assert_array_equal(np.asarray(gradient)[0, 0, 1], 0)

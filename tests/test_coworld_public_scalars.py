"""Training and wire contracts for the public scoreboard input experiment."""

import numpy as np
import pytest

from integrations.puffer_codec import encode_coworld_directional_observation
from integrations.softmax.neural_codec import encode_wire_observation, training_observation


@pytest.mark.parametrize("height,width", [(18, 21), (21, 18), (19, 20), (21, 21)])
def test_public_scalar_wire_training_prefix_and_ablation(height, width, tmp_path):
    pytest.importorskip("metta_training", reason="Optional private training adapter; exercised with pinned framework")
    from metta_training.environment import EnvironmentContext
    from integrations.metta_puffer import GeneralsPufferEnvironment

    kinds = np.ones((height, width), np.int32)
    owners = np.zeros_like(kinds)
    armies = np.zeros_like(kinds)
    kinds[2, 2], owners[2, 2], armies[2, 2] = 4, 1, 20
    owners[2, 3], armies[2, 3] = 1, 10
    kinds[-3, -3], owners[-3, -3], armies[-3, -3] = 4, 2, 20
    message = dict(height=height, width=width, type_grid=kinds.tolist(),
                   owner_grid=owners.tolist(), army_grid=armies.tolist(),
                   my_land=2, my_army=30, opp_land=50, opp_army=5000, turn=1000)
    observation = training_observation(message)
    old, old_mask = encode_wire_observation(message, directional=True, factorized_actions=False)
    rich, mask = encode_wire_observation(
        message, directional=True, factorized_actions=False, public_scalar_features=True,
    )
    assert rich.shape == (7056,) and mask.shape == (3529,)
    np.testing.assert_array_equal(rich[:4851], old)
    np.testing.assert_array_equal(mask, old_mask)
    expected = np.array([.5, 2/441, 50/441, np.log1p(30)/8, np.log1p(5000)/8], np.float32)
    np.testing.assert_allclose(rich.reshape(16, 441)[11:], np.broadcast_to(expected[:, None], (5, 441)), rtol=1e-6)
    direct, direct_mask = encode_coworld_directional_observation(
        observation, factorized_actions=False, public_scalar_features=True,
    )
    np.testing.assert_allclose(rich, direct, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(mask, direct_mask)
    zero, zero_mask = encode_wire_observation(
        message, directional=True, factorized_actions=False,
        public_scalar_features=True, public_scalar_ablation=True,
    )
    np.testing.assert_array_equal(zero[:4851], old)
    np.testing.assert_array_equal(zero[4851:], 0)
    np.testing.assert_array_equal(zero_mask, mask)
    env = GeneralsPufferEnvironment(
        context=EnvironmentContext(seed=9132, index=0, mode="train", output=tmp_path),
        coworld_classic=True, compact_features=True,
        lean_features=True, directional_features=True, public_scalar_features=True,
        factorized_actions=False, teacher=None,
    )
    assert env.spec.observation_size == 7056 and env.env.coworld_classic_rules
    encoded, env_mask = env._encode(observation)
    np.testing.assert_allclose(encoded, rich, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(env_mask, mask)


def test_scalar_flags_reject_incompatible_layout():
    with pytest.raises(ValueError):
        encode_coworld_directional_observation(None, public_scalar_ablation=True)
    with pytest.raises(ValueError):
        encode_coworld_directional_observation(None, public_scalar_features=True, include_timestep=True)

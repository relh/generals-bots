"""Checkpoint loading and legal recurrent inference for the pinned native model."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from generals import GeneralsEnv
from generals.core import game
from integrations.puffer_codec import encode_observation
from integrations.puffer_policy import PUFFER_REVISION, NativePufferPolicy


def test_native_checkpoint_produces_legal_action_and_resets_state(tmp_path: Path):
    board_size, hidden_size, layers = 6, 8, 2
    observation_size, action_size = 14 * board_size**2, 8 * board_size**2 + 1
    words = hidden_size * observation_size + (action_size + 1) * hidden_size + layers * 3 * hidden_size**2
    checkpoint = tmp_path / "checkpoint.bin"
    np.random.default_rng(73).normal(0, 0.02, words).astype(np.float32).tofile(checkpoint)
    policy = NativePufferPolicy(checkpoint, board_size=board_size, hidden_size=hidden_size, num_layers=layers)
    environment = GeneralsEnv(grid_dims=(board_size, board_size), truncation=20)
    state = environment.init_state(jax.random.PRNGKey(17))
    observation = game.get_observation(state, 0)
    logits, value = policy.predict(observation)
    action = np.asarray(policy.act(observation, jax.random.PRNGKey(1)))
    assert logits.shape == (action_size,)
    assert np.isfinite(value)
    assert action.shape == (5,)
    if action[0] == 0:
        position = action[1] * board_size + action[2]
        index = action[4] * 4 * board_size**2 + action[3] * board_size**2 + position
        assert bool(encode_observation(observation)[1][index])
    policy.reset()
    first_logits, first_value = policy.predict(observation)
    np.testing.assert_allclose(first_logits, logits)
    assert first_value == pytest.approx(value)


def test_from_run_checks_contract_and_checkpoint_size(tmp_path: Path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "final.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"invalid")
    record = {
        "build": {
            "revision": PUFFER_REVISION,
            "config": {
                "environment": "metta_generals",
                "precision": "float32",
                "fabric": None,
                "python_environment": {
                    "factory": "integrations.metta_puffer:GeneralsPufferEnvironment",
                    "options": {"board_size": 6},
                    "spec": {"observation_size": 504, "action_sizes": [289]},
                },
            },
        },
        "config": {"overrides": {"policy.hidden_size": 8, "policy.num_layers": 2}},
    }
    (run / "run.json").write_text(json.dumps(record))
    (run / "completed.json").write_text(json.dumps({"final_checkpoint": "checkpoints/final.bin"}))
    with pytest.raises(ValueError, match="finite float32"):
        NativePufferPolicy.from_run(run)
    record["build"]["config"]["python_environment"]["spec"]["observation_size"] = 503
    (run / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contract differs"):
        NativePufferPolicy.from_run(run)

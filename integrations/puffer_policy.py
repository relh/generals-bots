"""CPU inference for the pinned native PufferLib Generals checkpoint format.

The policy keeps recurrent state for one game. Call ``reset`` before each game.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.puffer_codec import decode_action, encode_observation

PUFFER_REVISION = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1 + exponent)
    return result


class NativePufferPolicy:
    """The pinned PufferNet linear encoder, MinGRU stack, and linear decoder."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        board_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 4,
    ) -> None:
        if min(board_size, hidden_size, num_layers) < 1:
            raise ValueError("Policy dimensions must be positive")
        self.board_size = board_size
        self.action_size = 8 * board_size**2 + 1
        observation_size = 14 * board_size**2
        words = hidden_size * observation_size + (self.action_size + 1) * hidden_size
        words += num_layers * 3 * hidden_size**2
        parameters = np.fromfile(checkpoint, dtype="<f4")
        if parameters.size != words or not np.isfinite(parameters).all():
            raise ValueError(f"Expected {words} finite float32 checkpoint parameters")
        offset = 0
        encoder_words = hidden_size * observation_size
        self.encoder = parameters[offset : offset + encoder_words].reshape(hidden_size, observation_size)
        offset += encoder_words
        decoder_words = (self.action_size + 1) * hidden_size
        self.decoder = parameters[offset : offset + decoder_words].reshape(self.action_size + 1, hidden_size)
        offset += decoder_words
        layer_words = 3 * hidden_size**2
        self.layers = [
            parameters[offset + i * layer_words : offset + (i + 1) * layer_words].reshape(3 * hidden_size, hidden_size)
            for i in range(num_layers)
        ]
        self.state = np.zeros((num_layers, hidden_size), dtype=np.float32)

    @classmethod
    def from_run(cls, run: Path, checkpoint: Path | None = None) -> "NativePufferPolicy":
        run = run.resolve()
        record = json.loads((run / "run.json").read_text())
        build = record["build"]
        config = build["config"]
        environment = config.get("python_environment")
        if (
            build["revision"] != PUFFER_REVISION
            or config["environment"] != "metta_generals"
            or config["precision"] != "float32"
            or config.get("fabric") is not None
            or environment is None
            or environment["factory"] != "integrations.metta_puffer:GeneralsPufferEnvironment"
        ):
            raise ValueError("Run does not use the supported native Generals Puffer policy")
        board_size = environment["options"].get("board_size", 10)
        spec = environment["spec"]
        if spec["observation_size"] != 14 * board_size**2 or spec["action_sizes"] != [8 * board_size**2 + 1]:
            raise ValueError("Run observation or action contract differs from the Generals adapter")
        if checkpoint is None:
            checkpoint = run / json.loads((run / "completed.json").read_text())["final_checkpoint"]
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_relative_to(run / "checkpoints"):
            raise ValueError("Checkpoint must belong to the training run")
        overrides = record["config"].get("overrides", {})
        return cls(
            checkpoint,
            board_size=board_size,
            hidden_size=overrides.get("policy.hidden_size", 128),
            num_layers=overrides.get("policy.num_layers", 4),
        )

    def reset(self) -> None:
        self.state.fill(0)

    def predict(self, observation) -> tuple[np.ndarray, float]:
        values, mask = encode_observation(observation)
        vector = np.asarray(values, dtype=np.float32)
        legal = np.asarray(mask, dtype=bool)
        if vector.shape != (self.encoder.shape[1],) or legal.shape != (self.action_size,):
            raise ValueError("Observation dimensions differ from the checkpoint")
        hidden_size = self.state.shape[1]
        vector = self.encoder @ vector
        for index, weights in enumerate(self.layers):
            combined = weights @ vector
            hidden = combined[:hidden_size]
            update = _sigmoid(combined[hidden_size : 2 * hidden_size])
            shortcut = _sigmoid(combined[2 * hidden_size :])
            candidate = np.where(hidden >= 0, hidden + 0.5, _sigmoid(hidden))
            current = self.state[index]
            next_state = current + update * (candidate - current)
            self.state[index] = next_state
            vector = shortcut * next_state + (1 - shortcut) * vector
        outputs = self.decoder @ vector
        logits = np.where(legal, outputs[:-1], -1e9)
        return logits, float(outputs[-1])

    def act(self, observation, key: jax.Array | None = None, *, deterministic: bool = True) -> jax.Array:
        logits, _ = self.predict(observation)
        if deterministic:
            index = int(np.argmax(logits))
        else:
            if key is None:
                raise ValueError("Stochastic inference requires a JAX key")
            index = int(jax.random.categorical(key, jnp.asarray(logits)))
        return decode_action(index, self.board_size)

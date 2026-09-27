"""Frozen float32 MinGRU inference for the pinned default Puffer5 architecture.

Matches encoder/decoder and mingru_gate in upstream 6ffa5b10 src/algo.cu.
This deliberately accepts only the verified Classic native build contract.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class NativePufferPolicy:
    def __init__(self, build: Path, training: Path, checkpoint: Path, sha256: str):
        manifest = json.loads(build.read_text())
        record = json.loads(training.read_text())
        if manifest != record["build"]:
            raise ValueError("Training/build manifests differ")
        config = manifest["config"]
        spec = config["python_environment"]["spec"]
        if (manifest["revision"] != "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
                or config["fabric"] is not None or config["precision"] != "float32"
                or spec["observation_size"] != 6174 or spec["action_sizes"] != [1765, 2]
                or spec["teacher"] or spec["routing"] or spec["replay_metadata_size"]):
            raise ValueError("Unsupported native policy contract")
        self.hidden = int(record["config"]["overrides"]["policy.hidden_size"])
        self.layers = int(record["config"]["overrides"]["policy.num_layers"])
        if self.hidden <= 0 or self.layers <= 0:
            raise ValueError("Invalid recurrent dimensions")
        data = checkpoint.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("Checkpoint SHA256 mismatch")
        offset = 0

        def take(shape):
            nonlocal offset
            # Puffer Allocator aligns the start of each registered tensor to16 bytes.
            offset = (offset + 15) & ~15
            count = int(np.prod(shape))
            array = np.frombuffer(data, dtype="<f4", count=count, offset=offset).reshape(shape)
            offset += count * 4
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite checkpoint parameters")
            return jnp.asarray(array)

        self.encoder = take((self.hidden, 6174))
        self.decoder = take((1768, self.hidden))
        self.recurrent = tuple(take((3 * self.hidden, self.hidden)) for _ in range(self.layers))
        if offset != len(data):
            raise ValueError("Unexpected checkpoint length")
        self.forward = jax.jit(self._forward)

    def initial_state(self, lanes):
        return jnp.zeros((self.layers, lanes, self.hidden), dtype=jnp.float32)

    def _forward(self, observations, state):
        x = jnp.matmul(observations, self.encoder.T, precision=jax.lax.Precision.HIGHEST)
        states = []
        for layer, weight in enumerate(self.recurrent):
            combined = jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST)
            hidden, gate, projection = jnp.split(combined, 3, axis=-1)
            z = jax.nn.sigmoid(gate)
            candidate = jnp.where(hidden >= 0, hidden + .5, jax.nn.sigmoid(hidden))
            difference = candidate - state[layer]
            # Match native lerp's stable branches.
            next_state = jnp.where(z < .5, state[layer] + z * difference,
                                   candidate - difference * (1 - z))
            s = jax.nn.sigmoid(projection)
            x = s * next_state + (1 - s) * x
            states.append(next_state)
        return jnp.matmul(x, self.decoder.T, precision=jax.lax.Precision.HIGHEST), jnp.stack(states)

    def actions(self, observations, masks, state):
        if observations.shape != (state.shape[1], 6174) or masks.shape != (state.shape[1], 1767):
            raise ValueError("Unexpected observation/mask dimensions")
        output, state = self.forward(jnp.asarray(observations, dtype=jnp.float32), state)
        logits = jnp.where(jnp.asarray(masks, dtype=bool), output[:, :1767], -jnp.inf)
        actions = jnp.stack((jnp.argmax(logits[:, :1765], axis=-1),
                             jnp.argmax(logits[:, 1765:], axis=-1)), axis=-1)
        return actions, state

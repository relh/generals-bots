"""NumPy inference for an exported, verified memoryless spatial checkpoint."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np


class SpatialPlayerPolicy:
    def __init__(self, bundle):
        manifest = json.loads((bundle / "spatial-policy.json").read_text())
        if manifest["schema"] != "puffer5-generals-spatial-v1":
            raise ValueError("Unsupported spatial policy bundle")
        if set(manifest["files"]) != {"build.json", "training.json", "policy.bin", "weights.npz"}:
            raise ValueError("Unexpected spatial policy bundle files")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Spatial bundle checksum mismatch: " + name)
        build = json.loads((bundle / "build.json").read_text())
        training = json.loads((bundle / "training.json").read_text())
        if training["build"] != build:
            raise ValueError("Spatial training/build manifests differ")
        config = build["config"]["fabric"]
        self.channels = config["options"]["channels"]
        self.observation_size = 441 * self.channels
        if self.channels not in (11, 12, 16) or config["observation_size"] != self.observation_size or config["action_sizes"] != [3529]:
            raise ValueError("Spatial bundle requires the public Classic observation and action contract")
        codec = build["config"]["python_environment"]["options"]
        if bool(codec.get("public_scalar_features")) != (self.channels == 16):
            raise ValueError("Spatial model channels differ from its public scalar codec")
        if manifest.get("channels", self.channels) != self.channels:
            raise ValueError("Spatial bundle channel metadata differs from its model")
        self.public_scalar_ablation = codec.get("public_scalar_ablation", False)
        if self.public_scalar_ablation and self.channels != 16:
            raise ValueError("Public scalar ablation requires the sixteen-channel codec")
        with np.load(bundle / "weights.npz", allow_pickle=False) as data:
            self.weights = {name: data[name].copy() for name in data.files}
        f, g = manifest["features"], manifest["global_features"]
        shapes = dict(input_kernel=(self.channels, f), context_kernel=(3, 3, f, f),
                      local_weight=(f,), local_bias=(f,), context_weight=(f,), context_bias=(f,),
                      global_kernel=(441 * f, g), global_weight=(g,), global_bias=(g,),
                      readout_kernel=(g, 3530), action_kernel=(f, 8),
                      output_weight=(3530,), output_bias=(3530,))
        self.prior_count = manifest["prior_count"]
        for i in range(self.prior_count):
            shapes[f"prior_source_{i}"] = (3530,)
            shapes[f"prior_weight_{i}"] = (3530,)
        if set(shapes) != set(self.weights):
            raise ValueError("Spatial tensor names differ from the declared architecture")
        for name, shape in shapes.items():
            value = self.weights[name]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid spatial tensor: " + name)
            if name.startswith("prior_source_"):
                if not np.issubdtype(value.dtype, np.integer) or np.any((value < 0) | (value >= self.observation_size)):
                    raise ValueError("Spatial prior source exceeds the observation")
            elif value.dtype != np.float32:
                raise ValueError("Spatial inference requires float32 weights")
        self.features = f

    @staticmethod
    def silu(value, xp=np):
        exponent = xp.exp(-xp.abs(value))
        sigmoid = xp.where(value >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
        return value * sigmoid

    def forward(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.observation_size or not np.isfinite(observations).all():
            raise ValueError("Invalid spatial observations")
        output = self._forward(observations, np)
        if not np.isfinite(output).all():
            raise FloatingPointError("Spatial inference produced nonfinite outputs")
        return output

    def _forward(self, observations, xp):
        w = self.weights
        if self.public_scalar_ablation:
            observations = xp.concatenate((observations[:, :4851], xp.zeros_like(observations[:, 4851:])), axis=1)
        obs = observations.reshape(-1, self.channels, 21, 21).transpose(0, 2, 3, 1)
        local = self.silu((obs @ w["input_kernel"]) * w["local_weight"] + w["local_bias"], xp)
        padded = xp.pad(local, ((0, 0), (1, 1), (1, 1), (0, 0)))
        context = xp.zeros_like(local)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
            context += padded[:, dy:dy + 21, dx:dx + 21] @ w["context_kernel"][dy, dx]
        context = self.silu(context * w["context_weight"] + w["context_bias"], xp)
        global_values = self.silu((context.reshape(-1, 441 * self.features) @ w["global_kernel"])
                                  * w["global_weight"] + w["global_bias"], xp)
        output = global_values @ w["readout_kernel"]
        action = context.reshape(-1, 441, self.features) @ w["action_kernel"]
        output = output + xp.concatenate((action.transpose(0, 2, 1).reshape(-1, 3528),
                                          xp.zeros((observations.shape[0], 2), dtype=observations.dtype)), axis=1)
        for i in range(self.prior_count):
            output += observations[:, w[f"prior_source_{i}"]] * w[f"prior_weight_{i}"]
        output = output * w["output_weight"] + w["output_bias"]
        return output

    def reset(self, seed):
        # The verified actor has no temporal dependence.
        pass

    def predict(self, seat, observation):
        values = np.asarray(observation.values, np.float32)
        mask = np.asarray(observation.action_masks, bool)
        if seat != 0 or values.shape != (1, self.observation_size) or mask.shape != (1, 3529) or not mask.any():
            raise ValueError("Invalid spatial single-seat observation or mask")
        output = self.forward(values)[0, :3529]
        logits = np.where(mask[0], output, -np.inf)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return SimpleNamespace(probabilities=probabilities.tolist())

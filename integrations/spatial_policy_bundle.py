"""NumPy inference for an exported, verified memoryless spatial checkpoint."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from integrations.spatial_action_sampling import (acting_logits, public_doomed_attack_route_penalty,
                                                  public_early_route_temperature, public_neutral_route_bonus,
                                                  public_weak_owned_route_penalty)


def structured_action_probabilities(outputs, legal, move_temperature, split_temperature,
                                    *, observations=None, neutral_route_bias=0.0,
                                    weak_owned_route_penalty=0.0, doomed_attack_route_penalty=0.0):
    """Match the native rollout categorical on the legal flat action set."""
    outputs = np.asarray(outputs, dtype=np.float32)
    legal = np.asarray(legal, dtype=bool)
    if outputs.shape != (3530,) or legal.shape != (3529,) or not legal.any():
        raise ValueError("Invalid spatial logits or legal-action mask")
    if not np.isfinite(outputs).all() or not all(
        np.isfinite(value) and value > 0 for value in (move_temperature, split_temperature)
    ):
        raise ValueError("Invalid spatial logits or action temperatures")
    if not np.isfinite(neutral_route_bias) or neutral_route_bias < 0 or (
            neutral_route_bias and observations is None):
        raise ValueError("Neutral route bias requires public observations and a finite nonnegative weight")
    if not np.isfinite(weak_owned_route_penalty) or weak_owned_route_penalty < 0 or (
            weak_owned_route_penalty and observations is None):
        raise ValueError("Weak owned route penalty requires public observations and a finite nonnegative weight")
    if not np.isfinite(doomed_attack_route_penalty) or doomed_attack_route_penalty < 0 or (
            doomed_attack_route_penalty and observations is None):
        raise ValueError("Doomed attack route penalty requires public observations and a finite nonnegative weight")
    transformed = acting_logits(outputs, move_temperature, split_temperature, np)[:3529]
    if neutral_route_bias or weak_owned_route_penalty or doomed_attack_route_penalty:
        public = np.asarray(observations, dtype=np.float32)
        if public.shape not in ((4851,), (5292,), (7056,)) or not np.isfinite(public).all():
            raise ValueError("Route adjustment requires one finite public observation")
        if neutral_route_bias:
            transformed += public_neutral_route_bonus(public, neutral_route_bias, np)
        if weak_owned_route_penalty:
            transformed += public_weak_owned_route_penalty(public, weak_owned_route_penalty, np)
        if doomed_attack_route_penalty:
            transformed += public_doomed_attack_route_penalty(public, doomed_attack_route_penalty, np)
    logits = np.where(legal, transformed, -np.inf)
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()


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
        acting = manifest.get("serving_action_selection", {"mode": "argmax"})
        if not isinstance(acting, dict):
            raise ValueError("Invalid spatial serving action selection")
        required = {"mode", "move_temperature", "split_temperature"}
        allowed = required | {"neutral_route_bias", "weak_owned_route_penalty", "doomed_attack_route_penalty",
                              "early_route_temperature", "early_route_turns"}
        if acting.get("mode") == "structured_sample" and required <= set(acting) <= allowed and all(
            isinstance(acting[key], (int, float)) and not isinstance(acting[key], bool)
                  and np.isfinite(acting[key]) and acting[key] > 0 for key in (
            "move_temperature", "split_temperature"
        )) and isinstance(acting.get("neutral_route_bias", 0.0), (int, float)) and not isinstance(
            acting.get("neutral_route_bias", 0.0), bool
        ) and np.isfinite(acting.get("neutral_route_bias", 0.0)) and acting.get("neutral_route_bias", 0.0) >= 0 and isinstance(
            acting.get("weak_owned_route_penalty", 0.0), (int, float)
        ) and not isinstance(acting.get("weak_owned_route_penalty", 0.0), bool) and np.isfinite(
            acting.get("weak_owned_route_penalty", 0.0)
        ) and acting.get("weak_owned_route_penalty", 0.0) >= 0 and isinstance(
            acting.get("doomed_attack_route_penalty", 0.0), (int, float)
        ) and not isinstance(acting.get("doomed_attack_route_penalty", 0.0), bool) and np.isfinite(
            acting.get("doomed_attack_route_penalty", 0.0)
        ) and acting.get("doomed_attack_route_penalty", 0.0) >= 0:
            self.action_mode = "structured_sample"
            self.move_temperature = float(acting["move_temperature"])
            self.split_temperature = float(acting["split_temperature"])
            self.neutral_route_bias = float(acting.get("neutral_route_bias", 0.0))
            self.weak_owned_route_penalty = float(acting.get("weak_owned_route_penalty", 0.0))
            self.doomed_attack_route_penalty = float(acting.get("doomed_attack_route_penalty", 0.0))
        elif acting == {"mode": "argmax"}:
            self.action_mode = "argmax"
            self.neutral_route_bias = 0.0
            self.weak_owned_route_penalty = 0.0
            self.doomed_attack_route_penalty = 0.0
        else:
            raise ValueError("Invalid spatial serving action selection")
        self.public_scalar_ablation = codec.get("public_scalar_ablation", False)
        if self.public_scalar_ablation and self.channels != 16:
            raise ValueError("Public scalar ablation requires the sixteen-channel codec")
        self.early_route_temperature = None
        self.early_route_turns = None
        if self.action_mode == "structured_sample":
            early_temperature = acting.get("early_route_temperature")
            early_turns = acting.get("early_route_turns")
            if (early_temperature is None) != (early_turns is None):
                raise ValueError("Early route temperature and turns must be paired")
            if early_temperature is not None:
                if self.public_scalar_ablation or self.channels != 16:
                    raise ValueError("Early route schedule requires full public scalar features")
                public_early_route_temperature(np.zeros((1, 16 * 441), np.float32), self.move_temperature,
                                               early_temperature, early_turns, np)
                self.early_route_temperature = float(early_temperature)
                self.early_route_turns = int(early_turns)
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
        self.reset("spatial-policy-default")

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
        # The graph has no temporal state. Sampling uses a reproducible stream.
        if self.action_mode == "structured_sample":
            digest = hashlib.sha256(str(seed).encode()).digest()
            self.action_rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))

    def predict(self, seat, observation):
        values = np.asarray(observation.values, np.float32)
        mask = np.asarray(observation.action_masks, bool)
        if seat != 0 or values.shape != (1, self.observation_size) or mask.shape != (1, 3529) or not mask.any():
            raise ValueError("Invalid spatial single-seat observation or mask")
        output = self.forward(values)[0]
        if self.action_mode == "structured_sample":
            move_temperature = self.move_temperature
            if self.early_route_temperature is not None:
                move_temperature = float(public_early_route_temperature(
                    values, move_temperature, self.early_route_temperature, self.early_route_turns, np)[0, 0])
            probabilities = structured_action_probabilities(
                output, mask[0], move_temperature, self.split_temperature,
                observations=values[0], neutral_route_bias=self.neutral_route_bias,
                weak_owned_route_penalty=self.weak_owned_route_penalty,
                doomed_attack_route_penalty=self.doomed_attack_route_penalty,
            )
        else:
            logits = np.where(mask[0], output[:3529], -np.inf)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
        return SimpleNamespace(probabilities=probabilities.tolist())

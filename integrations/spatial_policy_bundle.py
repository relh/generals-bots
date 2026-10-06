"""NumPy inference for an exported, verified memoryless spatial checkpoint."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np

from integrations.spatial_action_sampling import (
    acting_logits,
    public_doomed_attack_route_penalty,
    public_general_garrison_split_bias,
    public_early_route_temperature,
    public_neutral_route_bonus,
    public_weak_owned_route_penalty,
)


def structured_action_probabilities(outputs, legal, move_temperature, split_temperature,
                                    *, observations=None, neutral_route_bias=0.0,
                                    weak_owned_route_penalty=0.0, doomed_attack_route_penalty=0.0,
                                    route_half_weight=0.0, full_action_temperature=1.0, log_gap_scale=0.0,
                                    general_garrison_split_bias=0.0):
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
    if not np.isfinite(general_garrison_split_bias) or general_garrison_split_bias < 0 or (
            general_garrison_split_bias and observations is None):
        raise ValueError("General garrison bias requires public observations and a finite nonnegative weight")
    public = None
    if neutral_route_bias or weak_owned_route_penalty or doomed_attack_route_penalty or general_garrison_split_bias:
        public = np.asarray(observations, dtype=np.float32)
        if public.shape != (7056,) or not np.isfinite(public).all():
            raise ValueError("Route adjustment requires one finite public observation")
    split_bias = (public_general_garrison_split_bias(
        public, general_garrison_split_bias, np)
        if general_garrison_split_bias else None)
    transformed = acting_logits(outputs, move_temperature, split_temperature, np, split_bias,
                                route_half_weight=route_half_weight)[:3529]
    if neutral_route_bias or weak_owned_route_penalty or doomed_attack_route_penalty:
        if neutral_route_bias:
            transformed += public_neutral_route_bonus(public, neutral_route_bias, np)
        if weak_owned_route_penalty:
            transformed += public_weak_owned_route_penalty(public, weak_owned_route_penalty, np)
        if doomed_attack_route_penalty:
            transformed += public_doomed_attack_route_penalty(public, doomed_attack_route_penalty, np)
    from integrations.spatial_action_sampling import scale_action_logits

    transformed = scale_action_logits(transformed, full_action_temperature)
    from integrations.spatial_exploration import log_gap_logits, public_action_mask, validate_log_gap_scale

    validate_log_gap_scale(log_gap_scale)
    if log_gap_scale:
        public_mask = public_action_mask(np.asarray(observations), np)
        if not np.array_equal(public_mask, legal):
            raise ValueError("Public exploration mask differs from serving action mask")
        transformed = log_gap_logits(transformed, public_mask, log_gap_scale, np)
    logits = np.where(legal, transformed, -np.inf)
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()


class SpatialPlayerPolicy:
    def __init__(self, bundle):
        from integrations.native_spatial_asset import load_asset

        manifest = json.loads((bundle / "spatial-policy.json").read_text())
        if set(manifest) != {"schema", "files"} or manifest["schema"] != "generals-spatial-policy-v1":
            raise ValueError("Require the current spatial policy bundle")
        if set(manifest["files"]) != {"asset.json", "policy.bin", "weights.npz"}:
            raise ValueError("Unexpected spatial policy bundle files")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Spatial bundle checksum mismatch: " + name)
        self.asset = load_asset(bundle / "asset.json", manifest_sha256=manifest["files"]["asset.json"])
        if self.asset.learner is not None:
            raise ValueError("Serving bundles contain policy weights only")
        config = self.asset.metadata["fabric"]
        self.channels, self.observation_size = 16, 7056
        acting = self.asset.metadata["sampler"]
        required = {"mode", "move_temperature", "split_temperature"}
        optional = {"neutral_route_bias", "weak_owned_route_penalty", "doomed_attack_route_penalty",
                    "general_garrison_split_bias",
                    "early_route_temperature", "early_route_turns", "route_half_weight",
                    "full_action_temperature", "log_gap_scale"}
        if (not isinstance(acting, dict) or acting.get("mode") != "structured_sample"
                or not required <= set(acting) <= required | optional):
            raise ValueError("Spatial serving requires explicit structured_sample metadata")
        for key in ("move_temperature", "split_temperature"):
            value = acting[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError("Invalid spatial action temperature: " + key)
            setattr(self, key, float(value))
        for key in ("route_half_weight", "neutral_route_bias", "general_garrison_split_bias",
                    "weak_owned_route_penalty", "doomed_attack_route_penalty"):
            value = acting.get(key, 0.0)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                    or value < 0 or (key == "route_half_weight" and value > 1)):
                raise ValueError("Invalid spatial route sampler setting: " + key)
            setattr(self, key, float(value))
        from integrations.spatial_action_sampling import validate_full_action_temperature
        from integrations.spatial_exploration import validate_log_gap_scale

        self.full_action_temperature = validate_full_action_temperature(acting.get("full_action_temperature", 1.0))
        self.log_gap_scale = validate_log_gap_scale(acting.get("log_gap_scale", 0.0))
        self.action_mode = "structured_sample"
        self.early_route_temperature = acting.get("early_route_temperature")
        self.early_route_turns = acting.get("early_route_turns")
        if (self.early_route_temperature is None) != (self.early_route_turns is None):
            raise ValueError("Early route temperature and turns must be paired")
        if self.early_route_temperature is not None:
            public_early_route_temperature(np.zeros((1, 7056), np.float32), self.move_temperature,
                                           self.early_route_temperature, self.early_route_turns, np)
            self.early_route_temperature = float(self.early_route_temperature)
            self.early_route_turns = int(self.early_route_turns)
        with np.load(bundle / "weights.npz", allow_pickle=False) as data:
            self.weights = {name: data[name].copy() for name in data.files}
        f, g = config["options"]["features_per_site"], config["options"]["global_features"]
        from integrations.spatial_context_geometry import context_offsets

        radius = config["options"]["context_radius"]
        offsets = context_offsets(radius)
        extent = int(radius)
        shapes = dict(input_kernel=(self.channels, f), context_kernel=(2 * extent + 1, 2 * extent + 1, f, f),
                      local_weight=(f,), local_bias=(f,), context_weight=(f,), context_bias=(f,),
                      global_kernel=(441 * f, g), global_weight=(g,), global_bias=(g,),
                      readout_kernel=(g, 3530), action_kernel=(f, 8),
                      output_weight=(3530,), output_bias=(3530,))
        self.prior_count = 5
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
        active = {(y + extent, x + extent) for y, x in offsets}
        kernel = self.weights["context_kernel"]
        if any(np.any(kernel[y, x] != 0) for y in range(kernel.shape[0])
               for x in range(kernel.shape[1]) if (y, x) not in active):
            raise ValueError("Spatial context has weights outside its declared stencil")
        self.features = f
        self.reset("spatial-policy-default")

    @staticmethod
    def silu(value, xp=np):
        exponent = xp.exp(-xp.abs(value))
        sigmoid = xp.where(value >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
        return value * sigmoid

    def forward(self, observations):
        observations = np.asarray(observations, dtype=np.float32)
        if (observations.ndim != 2 or observations.shape[1] != self.observation_size
                or not np.isfinite(observations).all()):
            raise ValueError("Invalid spatial observations")
        output = self._forward(observations, np)
        if not np.isfinite(output).all():
            raise FloatingPointError("Spatial inference produced nonfinite outputs")
        return output

    def _forward(self, observations, xp, weights=None):
        w = self.weights if weights is None else weights
        obs = observations.reshape(-1, self.channels, 21, 21).transpose(0, 2, 3, 1)
        local = self.silu((obs @ w["input_kernel"]) * w["local_weight"] + w["local_bias"], xp)
        from integrations.spatial_context_geometry import kernel_offsets

        size = w["context_kernel"].shape[0]
        extent = size // 2
        padded = xp.pad(local, ((0, 0), (extent, extent), (extent, extent), (0, 0)))
        context = xp.zeros_like(local)
        for dy, dx in kernel_offsets(size):
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
        digest = hashlib.sha256(str(seed).encode()).digest()
        self.action_rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))

    def predict(self, seat, observation):
        values = np.asarray(observation.values, np.float32)
        mask = np.asarray(observation.action_masks, bool)
        if seat != 0 or values.shape != (1, self.observation_size) or mask.shape != (1, 3529) or not mask.any():
            raise ValueError("Invalid spatial single-seat observation or mask")
        output = self.forward(values)[0]
        move_temperature = self.move_temperature
        if self.early_route_temperature is not None:
            move_temperature = float(public_early_route_temperature(
                values, move_temperature, self.early_route_temperature, self.early_route_turns, np)[0, 0])
        probabilities = structured_action_probabilities(
            output, mask[0], move_temperature, self.split_temperature,
            observations=values[0], neutral_route_bias=self.neutral_route_bias,
            weak_owned_route_penalty=self.weak_owned_route_penalty,
            doomed_attack_route_penalty=self.doomed_attack_route_penalty,
            general_garrison_split_bias=self.general_garrison_split_bias,
            route_half_weight=self.route_half_weight, full_action_temperature=self.full_action_temperature,
            log_gap_scale=self.log_gap_scale,
        )
        return SimpleNamespace(probabilities=probabilities.tolist())

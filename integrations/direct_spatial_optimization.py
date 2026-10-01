"""Evaluate the pinned memoryless actor with convolutions and matrix products.

Build every weight lookup from Fabric's realized topology and sharing tables.
Puffer still owns PPO, parameter storage, and updates. Rollout uses the original
bridge unless METTA_DIRECT_SPATIAL_ROLLOUT=1 explicitly enables the same algebra.
"""

from dataclasses import dataclass
import functools
import hashlib
import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np

from integrations.memoryless_optimization import BRIDGE_SHA256, verify_configuration


@dataclass(frozen=True)
class DirectTape:
    parameters: object
    observations: object
    predictions: object

    def __getitem__(self, index):
        # Plain PPO's auxiliary collector only needs the observation batch axis.
        if index == 2:
            return self.observations.transpose(1, 2, 0)[..., None]
        if index == 4:
            return self.predictions
        raise IndexError(index)


class DirectSpatial:
    def __init__(self, policy):
        buffers = policy.buffers
        fn = buffers.fn
        pops = {p.name: p for p in fn.spec.pooled.populations}
        if set(pops) != {"input", "SiLU", "ContextSiLU", "GlobalSiLU", "Output"}:
            raise ValueError("Unexpected spatial populations")
        cells = 441
        features = pops["SiLU"].n // cells
        global_features = pops["GlobalSiLU"].n
        if pops["SiLU"].n != cells * features or pops["ContextSiLU"].n != cells * features:
            raise ValueError("Spatial population dimensions differ")
        self.features = features
        self.channels = pops["input"].n // cells
        self.observation_size = cells * self.channels
        if self.channels not in (11, 12, 16) or pops["input"].n != self.observation_size:
            raise ValueError("Unsupported directional spatial observation layout")
        self.global_features = global_features
        leaves = jax.tree_util.tree_flatten_with_path(fn.params(buffers.template))[0]
        layout = {
            tuple(key.key for key in path): spec
            for (path, _), spec in zip(leaves, buffers.parameters, strict=True)
        }
        specs = fn.spec.leaf_specs()

        def docs(pop):
            return np.arange(pop.n) if pop.document_index is None else np.asarray(pop.document_index)

        def atom(owner, field):
            pop = pops[owner]
            leaf = specs[("params", owner, field)]
            rows = np.arange(pop.n) if leaf.group_rows is None else np.asarray(leaf.group_rows)
            indices = np.empty(pop.n, np.int32)
            indices[docs(pop)] = layout[(owner, field)].offset + rows
            return indices

        self.local_weight = atom("SiLU", "weight").reshape(cells, features)
        self.local_bias = atom("SiLU", "bias").reshape(cells, features)
        self.context_weight = atom("ContextSiLU", "weight").reshape(cells, features)
        self.context_bias = atom("ContextSiLU", "bias").reshape(cells, features)
        for indices in (self.local_weight, self.local_bias, self.context_weight, self.context_bias):
            if not np.all(indices == indices[:1]):
                raise ValueError("Site atom parameters are not shared across cells")
        self.global_weight = atom("GlobalSiLU", "weight")
        self.global_bias = atom("GlobalSiLU", "bias")
        self.output_weight = atom("Output", "W")
        self.output_bias = atom("Output", "b")

        def edges(cls):
            if cls.delay or cls.into != "drive" or cls.src.rate != 1 or cls.dst.rate != 1:
                raise ValueError("Direct evaluation requires current-tick scalar edges")
            occ = buffers.template["occupancy"].get(cls.name)
            if occ is None:
                # Dense/stencil representations bake the original edge table.
                pairs = np.asarray(cls.edges)
                src, dst = docs(cls.src)[pairs[:, 0]], docs(cls.dst)[pairs[:, 1]]
                leaf = specs[("params", cls.name, "weight")]
                rows = np.arange(len(pairs)) if leaf.group_rows is None else np.asarray(leaf.group_rows)
            else:
                n = int(np.asarray(occ["live"]))
                src = docs(cls.src)[np.asarray(occ["send"])[:n]]
                dst = docs(cls.dst)[np.asarray(occ["recv"])[:n]]
                rows = np.asarray(occ["data"]["weight"])[:n]
            weight = layout[(cls.name, "weight")].offset + rows
            return src, dst, weight

        def shared(shape, keys, values):
            result = np.full(shape, -1, np.int32)
            for key, value in zip(zip(*keys), values, strict=True):
                if result[key] not in (-1, value):
                    raise ValueError("Expected convolution or site weights are not shared")
                result[key] = value
            return result

        def dense(shape, src, dst, weights):
            keys = np.ravel_multi_index((src, dst), shape)
            if len(keys) != np.prod(shape) or len(np.unique(keys)) != np.prod(shape):
                raise ValueError("Dense coupling is incomplete or contains repeated pairs")
            result = np.empty(shape, np.int32)
            result[src, dst] = weights
            return result

        self.priors = []
        seen = set()
        for cls in fn.spec.pooled.edge_classes:
            src, dst, weights = edges(cls)
            pair = cls.src.name, cls.dst.name
            if pair == ("input", "SiLU"):
                if not np.all(src % cells == dst // features):
                    raise ValueError("Input coupling reaches another site")
                self.input_kernel = shared((self.channels, features), (src // cells, dst % features), weights)
            elif pair == ("SiLU", "ContextSiLU"):
                dx = (src // features) % 21 - (dst // features) % 21
                dy = (src // features) // 21 - (dst // features) // 21
                if np.any(np.abs(dx) + np.abs(dy) > 1):
                    raise ValueError("Context exceeds the verified cross stencil")
                if len(src) != (5 * cells - 4 * 21) * features ** 2:
                    raise ValueError("Context cross stencil is incomplete")
                self.context_kernel = shared((3, 3, features, features),
                                             (dy + 1, dx + 1, src % features, dst % features), weights)
            elif pair == ("ContextSiLU", "Output"):
                if np.any(dst >= cells * 8) or not np.all(src // features == dst % cells):
                    raise ValueError("Action readout reaches another site or non-move output")
                self.action_kernel = shared((features, 8), (src % features, dst // cells), weights)
            elif pair == ("ContextSiLU", "GlobalSiLU"):
                self.global_kernel = dense((cells * features, global_features), src, dst, weights)
            elif pair == ("GlobalSiLU", "Output"):
                self.readout_kernel = dense((global_features, 3530), src, dst, weights)
            elif pair == ("input", "Output"):
                if len(np.unique(dst)) != len(dst):
                    raise ValueError("Public prior has multiple inputs per output")
                source_indices = np.zeros(3530, np.int32)
                weight_indices = np.full(3530, -1, np.int32)
                source_indices[dst] = src
                weight_indices[dst] = weights
                self.priors.append((source_indices, weight_indices))
            else:
                raise ValueError(f"Unexpected coupling: {pair}")
            seen.add(pair)
            if len(np.unique(np.stack((src, dst), axis=1), axis=0)) != len(src):
                raise ValueError("Spatial coupling contains repeated endpoint pairs")
        if seen != {("input", "SiLU"), ("SiLU", "ContextSiLU"),
                    ("ContextSiLU", "Output"), ("ContextSiLU", "GlobalSiLU"),
                    ("GlobalSiLU", "Output"), ("input", "Output")}:
            raise ValueError("Spatial topology is incomplete")
        if any(np.any(x < 0) for x in (self.input_kernel, self.action_kernel)):
            raise ValueError("Shared projection is incomplete")
        self.forward = jax.jit(self.evaluate)
        self.gradient = jax.jit(jax.grad(lambda p, obs, cot: jnp.sum(self.evaluate(p, obs) * cot)))

    @staticmethod
    def weights(parameters, indices):
        return jnp.where(indices >= 0, parameters[np.maximum(indices, 0)], 0)

    def evaluate(self, parameters, observations):
        shape = observations.shape[:-1]
        obs = observations.reshape(-1, self.channels, 21, 21).transpose(0, 2, 3, 1)
        local = jnp.matmul(obs, self.weights(parameters, self.input_kernel), precision=jax.lax.Precision.HIGHEST)
        local = jax.nn.silu(local * parameters[self.local_weight[0]] + parameters[self.local_bias[0]])
        context = jax.lax.conv_general_dilated(
            local, self.weights(parameters, self.context_kernel), (1, 1), "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=jax.lax.Precision.HIGHEST,
        )
        context = jax.nn.silu(context * parameters[self.context_weight[0]] + parameters[self.context_bias[0]])
        global_values = jnp.matmul(context.reshape(-1, 441 * self.features), parameters[self.global_kernel],
                                   precision=jax.lax.Precision.HIGHEST)
        global_values = jax.nn.silu(global_values * parameters[self.global_weight] + parameters[self.global_bias])
        outputs = jnp.matmul(global_values, parameters[self.readout_kernel], precision=jax.lax.Precision.HIGHEST)
        action = jnp.matmul(context.reshape(-1, 441, self.features), parameters[self.action_kernel],
                            precision=jax.lax.Precision.HIGHEST)
        outputs = outputs.at[:, :3528].add(action.transpose(0, 2, 1).reshape(-1, 3528))
        flat_obs = observations.reshape(-1, self.observation_size)
        for source, weights in self.priors:
            outputs = outputs + flat_obs[:, source] * self.weights(parameters, weights)
        outputs = outputs * parameters[self.output_weight] + parameters[self.output_bias]
        return outputs.reshape(*shape, 3530)


def install(native_module=None):
    native_module = native_module or importlib.import_module("metta_training.native_fabric")
    from pathlib import Path

    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Native bridge differs from the verified callback contract")
    cls = native_module.NativeFabricPolicy
    if getattr(cls, "_generals_direct_spatial", False) or getattr(cls, "_generals_optimization_rows", False):
        raise RuntimeError("A spatial optimization adapter is already installed")
    initialize, forward, backward = cls.__init__, cls._forward_arrays, cls.backward_device_arrays

    @functools.wraps(initialize)
    def init(self, configuration, *args, **kwargs):
        verify_configuration(configuration)
        initialize(self, configuration, *args, **kwargs)
        self.direct_spatial = DirectSpatial(self)
        self.spatial_optimizer_layout = os.environ.get("METTA_SPATIAL_OPTIMIZER_LAYOUT", "native")
        context_mode = os.environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX", "0")
        if context_mode not in ("0", "1") or (context_mode == "1" and (
                self.spatial_optimizer_layout != "logical"
                or os.environ.get("METTA_SPATIAL_MUON_DENSE_ORIENTATION") != "canonical")):
            raise ValueError("Convolution Muon requires logical layout and canonical dense scaling")
        if self.spatial_optimizer_layout == "logical":
            from integrations.spatial_optimizer_layout import logical_optimizer_shapes

            self.shapes, self.spatial_optimizer_layout_report = logical_optimizer_shapes(
                self.direct_spatial, self.buffers,
                context_matrix=context_mode == "1",
            )
        elif self.spatial_optimizer_layout != "native":
            raise ValueError("Spatial optimizer layout must be native or logical")
        self.direct_spatial_rollout = os.environ.get("METTA_DIRECT_SPATIAL_ROLLOUT") == "1"
        self.spatial_policy_temperature = float(os.environ.get("METTA_SPATIAL_POLICY_TEMPERATURE", "1"))
        if not np.isfinite(self.spatial_policy_temperature) or self.spatial_policy_temperature <= 0:
            raise ValueError("Spatial policy temperature must be finite and positive")
        split_temperature = os.environ.get("METTA_SPATIAL_SPLIT_TEMPERATURE")
        self.spatial_split_temperature = float(split_temperature) if split_temperature is not None else None
        if self.spatial_split_temperature is not None and (
                not np.isfinite(self.spatial_split_temperature) or self.spatial_split_temperature <= 0):
            raise ValueError("Spatial split temperature must be finite and positive")
        self.spatial_neutral_route_bias = float(os.environ.get("METTA_SPATIAL_NEUTRAL_ROUTE_BIAS", "0"))
        if not np.isfinite(self.spatial_neutral_route_bias) or self.spatial_neutral_route_bias < 0:
            raise ValueError("Neutral route bias must be finite and nonnegative")
        if self.spatial_neutral_route_bias and self.spatial_split_temperature is None:
            raise ValueError("Neutral route bias requires structured route and split sampling")
        if (self.spatial_policy_temperature != 1 or self.spatial_split_temperature is not None) and not self.direct_spatial_rollout:
            raise ValueError("Temperature requires identical direct rollout and optimization algebra")

    @functools.wraps(forward)
    def direct_forward(self, parameters, state, transported, terminals, batch, time, rollout):
        if rollout and not self.direct_spatial_rollout:
            return forward(self, parameters, state, transported, terminals, batch, time, rollout)
        if transported.shape != (batch, time, self.direct_spatial.observation_size) or terminals.shape != (batch, time):
            raise ValueError("Direct optimization requires plain public observations")
        outputs = self.direct_spatial.forward(parameters, transported)
        acting = outputs
        if self.spatial_split_temperature is not None:
            from integrations.spatial_action_sampling import acting_logits

            acting = acting_logits(outputs, self.spatial_policy_temperature,
                                  self.spatial_split_temperature, jnp)
        elif self.spatial_policy_temperature != 1:
            acting = outputs.at[..., :3529].divide(self.spatial_policy_temperature)
        if self.spatial_neutral_route_bias:
            from integrations.spatial_action_sampling import public_neutral_route_bonus

            bonus = public_neutral_route_bonus(transported, self.spatial_neutral_route_bias, jnp)
            acting = acting.at[..., :3529].add(bonus)
        if not bool(jnp.isfinite(acting).all()):
            raise FloatingPointError("Direct spatial predictions became nonfinite")
        return acting, state, DirectTape(parameters, transported, outputs)

    @functools.wraps(backward)
    def direct_backward(self, tape, logits, values):
        if not isinstance(tape, DirectTape):
            return backward(self, tape, logits, values)
        if logits.shape != tape.predictions.shape[:-1] + (3529,) or values.shape != tape.predictions.shape[:-1]:
            raise ValueError("Direct PPO cotangents differ from predictions")
        self.updates += 1
        self.active_objectives.clear()
        coefficient = self.teacher_phase.ppo_coefficient
        if self.spatial_split_temperature is not None:
            from integrations.spatial_action_sampling import raw_cotangents

            cotangents = raw_cotangents(tape.predictions, logits, values,
                                       self.spatial_policy_temperature,
                                       self.spatial_split_temperature, jnp) * coefficient
        else:
            cotangents = jnp.concatenate((logits / self.spatial_policy_temperature,
                                          values[..., None]), axis=-1) * coefficient
        gradient = self.direct_spatial.gradient(tape.parameters, tape.observations, cotangents)
        if not bool(jnp.isfinite(gradient).all()):
            raise FloatingPointError("Direct spatial gradients became nonfinite")
        return gradient

    cls.__init__ = init
    cls._forward_arrays = direct_forward
    cls.backward_device_arrays = direct_backward
    cls._generals_direct_spatial = True
    return forward, backward

"""Evaluate the pinned memoryless actor with convolutions and matrix products.

Build every weight lookup from Fabric's realized topology and sharing tables.
Puffer owns PPO, parameter storage, and updates. The supported native path uses
the same canonical spatial algebra for rollout and optimization.
"""

import functools
import hashlib
import importlib
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from integrations.spatial_context_geometry import context_offsets
from integrations.spatial_native_contract import BRIDGE_SHA256, verify_configuration


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
        self.channels = 16
        self.observation_size = 7056
        if pops["input"].n != self.observation_size:
            raise ValueError("Direct spatial optimization requires the canonical sixteen public planes")
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
                extent = int(max(np.abs(dx).max(), np.abs(dy).max()))
                offsets = context_offsets(extent + .01)
                actual = set(zip(dy.tolist(), dx.tolist()))
                if actual != set(offsets):
                    raise ValueError("Context differs from the supported stencil")
                expected = sum((21 - abs(y)) * (21 - abs(x)) for y, x in offsets)
                if len(src) != expected * features ** 2:
                    raise ValueError("Context stencil is incomplete")
                self.context_kernel = shared((2 * extent + 1, 2 * extent + 1, features, features),
                                             (dy + extent, dx + extent, src % features, dst % features), weights)
                if any(np.any(self.context_kernel[y + extent, x + extent] < 0) for y, x in offsets):
                    raise ValueError("Context feature projection is incomplete")
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


def compile_acting_transform(policy):
    """Compile validated sampler postprocessing separately from the raw model."""
    spatial_doomed_attack_route_penalty = policy.spatial_doomed_attack_route_penalty
    spatial_early_route_temperature = policy.spatial_early_route_temperature
    spatial_early_route_turns = policy.spatial_early_route_turns
    spatial_full_action_temperature = policy.spatial_full_action_temperature
    spatial_neutral_route_bias = policy.spatial_neutral_route_bias
    spatial_policy_temperature = policy.spatial_policy_temperature
    spatial_route_half_weight = policy.spatial_route_half_weight
    spatial_split_temperature = policy.spatial_split_temperature
    spatial_weak_owned_route_penalty = policy.spatial_weak_owned_route_penalty

    def transform(outputs, transported):
        move_temperature = spatial_policy_temperature
        if spatial_early_route_temperature is not None:
            from integrations.spatial_action_sampling import public_early_route_temperature

            move_temperature = public_early_route_temperature(
                transported, move_temperature, spatial_early_route_temperature,
                spatial_early_route_turns, jnp)
        from integrations.spatial_action_sampling import acting_logits

        acting = acting_logits(outputs, move_temperature, spatial_split_temperature, jnp,
                               route_half_weight=spatial_route_half_weight)
        if spatial_neutral_route_bias:
            from integrations.spatial_action_sampling import public_neutral_route_bonus

            bonus = public_neutral_route_bonus(transported, spatial_neutral_route_bias, jnp)
            acting = acting.at[..., :3529].add(bonus)
        if spatial_weak_owned_route_penalty:
            from integrations.spatial_action_sampling import public_weak_owned_route_penalty

            penalty = public_weak_owned_route_penalty(transported, spatial_weak_owned_route_penalty, jnp)
            acting = acting.at[..., :3529].add(penalty)
        if spatial_doomed_attack_route_penalty:
            from integrations.spatial_action_sampling import public_doomed_attack_route_penalty

            penalty = public_doomed_attack_route_penalty(transported, spatial_doomed_attack_route_penalty, jnp)
            acting = acting.at[..., :3529].add(penalty)
        if spatial_full_action_temperature != 1:
            acting = acting.at[..., :3529].divide(spatial_full_action_temperature)
        return acting, jnp.isfinite(acting).all()

    return jax.jit(transform)


def install(native_module=None):
    native_module = native_module or importlib.import_module("metta_training.native_fabric")
    from pathlib import Path

    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Native bridge differs from the verified callback contract")
    cls = native_module.NativeFabricPolicy
    if getattr(cls, "_generals_direct_spatial", False):
        raise RuntimeError("A spatial optimization adapter is already installed")
    initialize, forward, backward = cls.__init__, cls._forward_arrays, cls.backward_device_arrays

    @functools.wraps(initialize)
    def init(self, configuration, *args, **kwargs):
        verify_configuration(configuration)
        initialize(self, configuration, *args, **kwargs)
        self.direct_spatial = DirectSpatial(self)
        self.state_words = 0  # Compiler mailboxes remain internal; no external carry.
        if (os.environ.get("METTA_DIRECT_SPATIAL_ROLLOUT") != "1"
                or os.environ.get("METTA_SPATIAL_OPTIMIZER_LAYOUT") != "logical"
                or os.environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX") != "1"
                or os.environ.get("METTA_SPATIAL_MUON_DENSE_ORIENTATION") != "canonical"):
            raise ValueError("Spatial training requires direct rollout and canonical logical context Muon")
        from integrations.spatial_optimizer_layout import logical_optimizer_shapes

        self.shapes, self.spatial_optimizer_layout_report = logical_optimizer_shapes(
            self.direct_spatial, self.buffers, context_matrix=True,
        )
        self.spatial_policy_temperature = float(os.environ.get("METTA_SPATIAL_POLICY_TEMPERATURE", "1"))
        if not np.isfinite(self.spatial_policy_temperature) or self.spatial_policy_temperature <= 0:
            raise ValueError("Spatial policy temperature must be finite and positive")
        early_temperature = os.environ.get("METTA_SPATIAL_EARLY_ROUTE_TEMPERATURE")
        early_turns = os.environ.get("METTA_SPATIAL_EARLY_ROUTE_TURNS")
        if (early_temperature is None) != (early_turns is None):
            raise ValueError("Early route temperature and turns must be set together")
        self.spatial_early_route_temperature = float(early_temperature) if early_temperature is not None else None
        self.spatial_early_route_turns = int(early_turns) if early_turns is not None else None
        self.spatial_split_temperature = float(os.environ.get("METTA_SPATIAL_SPLIT_TEMPERATURE", "1"))
        if not np.isfinite(self.spatial_split_temperature) or self.spatial_split_temperature <= 0:
            raise ValueError("Spatial split temperature must be finite and positive")
        from integrations.spatial_action_sampling import validate_full_action_temperature

        self.spatial_full_action_temperature = validate_full_action_temperature(
            float(os.environ.get("METTA_SPATIAL_FULL_ACTION_TEMPERATURE", "1")))

        if "METTA_SPATIAL_LOG_GAP_SCALE" in os.environ:
            raise ValueError("Unsupported retired sampler environment: METTA_SPATIAL_LOG_GAP_SCALE")
        self.spatial_route_half_weight = float(os.environ.get("METTA_SPATIAL_ROUTE_HALF_WEIGHT", "0"))
        if (not np.isfinite(self.spatial_route_half_weight) or
                not 0 <= self.spatial_route_half_weight <= 1):
            raise ValueError("Route half weight requires structured sampling and must be between zero and one")
        if self.spatial_early_route_temperature is not None:
            from integrations.spatial_action_sampling import public_early_route_temperature

            public_early_route_temperature(np.zeros((1, 16 * 441), np.float32), self.spatial_policy_temperature,
                                           self.spatial_early_route_temperature, self.spatial_early_route_turns, np)
        self.spatial_neutral_route_bias = float(os.environ.get("METTA_SPATIAL_NEUTRAL_ROUTE_BIAS", "0"))
        if not np.isfinite(self.spatial_neutral_route_bias) or self.spatial_neutral_route_bias < 0:
            raise ValueError("Neutral route bias must be finite and nonnegative")
        self.spatial_weak_owned_route_penalty = float(os.environ.get("METTA_SPATIAL_WEAK_OWNED_ROUTE_PENALTY", "0"))
        if not np.isfinite(self.spatial_weak_owned_route_penalty) or self.spatial_weak_owned_route_penalty < 0:
            raise ValueError("Weak owned route penalty must be finite and nonnegative")
        self.spatial_doomed_attack_route_penalty = float(
            os.environ.get("METTA_SPATIAL_DOOMED_ATTACK_ROUTE_PENALTY", "0"))
        if not np.isfinite(self.spatial_doomed_attack_route_penalty) or self.spatial_doomed_attack_route_penalty < 0:
            raise ValueError("Doomed attack route penalty must be finite and nonnegative")
        self.spatial_acting_transform = compile_acting_transform(self)

    @functools.wraps(forward)
    def direct_forward(self, parameters, transported, terminals, batch, time, rollout):
        if transported.shape != (batch, time, self.direct_spatial.observation_size) or terminals.shape != (batch, time):
            raise ValueError("Direct optimization requires plain public observations")
        outputs = self.direct_spatial.forward(parameters, transported)
        acting, finite = self.spatial_acting_transform(outputs, transported)
        if not bool(finite):
            raise FloatingPointError("Direct spatial predictions became nonfinite")
        return acting, DirectTape(parameters, transported, outputs)

    @functools.wraps(backward)
    def direct_backward(self, tape, logits, values):
        if not isinstance(tape, DirectTape):
            raise TypeError("Direct PPO backward requires its canonical forward tape")
        if logits.shape != tape.predictions.shape[:-1] + (3529,) or values.shape != tape.predictions.shape[:-1]:
            raise ValueError("Direct PPO cotangents differ from predictions")
        self.updates += 1
        self.active_objectives.clear()
        coefficient = self.teacher_phase.ppo_coefficient
        # Chain rule for the final action-only transform; preserve value gradients.
        logits = logits / self.spatial_full_action_temperature
        from integrations.spatial_action_sampling import raw_cotangents

        move_temperature = self.spatial_policy_temperature
        if self.spatial_early_route_temperature is not None:
            from integrations.spatial_action_sampling import public_early_route_temperature

            move_temperature = public_early_route_temperature(
                tape.observations, move_temperature, self.spatial_early_route_temperature,
                self.spatial_early_route_turns, jnp)
        cotangents = raw_cotangents(tape.predictions, logits, values, move_temperature,
                                   self.spatial_split_temperature, jnp,
                                   route_half_weight=self.spatial_route_half_weight) * coefficient
        gradient = self.direct_spatial.gradient(tape.parameters, tape.observations, cotangents)
        if not bool(jnp.isfinite(gradient).all()):
            raise FloatingPointError("Direct spatial gradients became nonfinite")
        return gradient

    def device_forward(self, parameters, observations, terminals, batch, time, rollout):
        params = jax.dlpack.from_dlpack(parameters)
        observed = jax.dlpack.from_dlpack(observations)
        done = (jax.dlpack.from_dlpack(terminals) if terminals is not None else
                jnp.zeros((batch, time), dtype=jnp.float32, device=params.device))
        return self._forward_arrays(params, observed, done, batch, time, rollout)

    def host_forward(self, parameters, observations, terminals, batch, time, rollout):
        acting, tape = self._forward_arrays(
            jnp.asarray(np.frombuffer(parameters, dtype=np.float32)),
            jnp.asarray(np.frombuffer(observations, dtype=np.float32).reshape(batch,time,self.input_size)),
            jnp.asarray(np.frombuffer(terminals, dtype=np.float32).reshape(batch,time)),
            batch, time, rollout)
        return np.asarray(acting).tobytes(), tape

    cls.__init__ = init
    cls.forward_device = device_forward
    cls.forward = host_forward
    cls._forward_arrays = direct_forward
    cls.backward_device_arrays = direct_backward
    cls._generals_direct_spatial = True
    return forward, backward

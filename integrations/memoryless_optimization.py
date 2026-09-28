"""Flatten optimization sequences for the verified mailbox-only spatial actor.

Rollout and serving keep the original implementation. Puffer owns PPO and the
optimizer; this changes only how independent optimization rows reach Fabric.
"""

import functools
import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

FACTORY = "integrations.generals_fabric:two_stage_tied_local_action_policy"
FACTORY_SHA256 = "445724d7330622ca44a9f81ffb4531013add2596c8eb95ce6d93141fd196c322"
BRIDGE_SHA256 = "c1bed03201af5133badfe8c5fa1566efc3830acc73c68798fbf5b7f7d7e051c1"


@dataclass(frozen=True)
class OptimizationRows:
    inner: tuple
    batch: int
    time: int


def verify_configuration(configuration):
    config = json.loads(configuration)
    if config["factory"] != FACTORY:
        raise ValueError("Optimization flattening requires the archived spatial actor")
    module = importlib.import_module(FACTORY.split(":")[0])
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != FACTORY_SHA256:
        raise ValueError("Spatial actor source differs from the mailbox-only proof")
    if config["observation_size"] != 4851 or config["action_sizes"] != [3529]:
        raise ValueError("Optimization flattening requires the public flat Classic codec")
    if config.get("compiler") != "standard" or any(
        config["options"].get(key) != value for key, value in (("channels", 11), ("height", 21), ("width", 21))
    ):
        raise ValueError("Optimization flattening requires the verified compiler and board layout")
    if config["options"].get("broadcast_global_context", False):
        raise ValueError("Global feedback requires a separate temporal-dependence audit")
    forbidden = (
        "teacher", "self_distillation", "ema_prior", "horde", "rnd",
        "group_returns", "quantile_critic", "retrace", "routing", "train_mask_column",
    )
    if any(config.get(key) is not None for key in forbidden):
        raise ValueError("Auxiliary or temporal objectives are not supported by this optimization")
    if config.get("losses") or config.get("replay_metadata_size", 0):
        raise ValueError("Optimization flattening requires plain PPO without replay metadata")


def install(native_module=None):
    """Install in the embedded trainer process; return originals for numerical audit."""
    if native_module is None:
        native_module = importlib.import_module("metta_training.native_fabric")
    if hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Native bridge differs from the audited device callback implementation")
    NativeFabricPolicy = native_module.NativeFabricPolicy

    if getattr(NativeFabricPolicy, "_generals_optimization_rows", False):
        raise RuntimeError("Optimization row adapter is already installed")
    original_init = NativeFabricPolicy.__init__
    original_forward = NativeFabricPolicy._forward_arrays
    original_backward = NativeFabricPolicy.backward_device_arrays
    original_critic = NativeFabricPolicy.critic_predictions

    @functools.wraps(original_init)
    def initialize(self, configuration, *args, **kwargs):
        verify_configuration(configuration)
        original_init(self, configuration, *args, **kwargs)

    @functools.wraps(original_forward)
    def forward(self, parameters, state, transported, terminals, batch, time, rollout):
        if rollout or time == 1:
            return original_forward(self, parameters, state, transported, terminals, batch, time, rollout)
        if transported.shape[:2] != (batch, time) or terminals.shape != (batch, time):
            raise ValueError("Optimization sequence dimensions differ from the declared batch")
        import jax.numpy as jnp

        # All actors/couplings are rate-one mailboxes with no global feedback.
        # Each row therefore has within-step credit and no temporal dependency.
        rows = batch * time
        outputs, _, inner = original_forward(
            self, parameters, jnp.repeat(state, time, axis=0),
            transported.reshape(rows, 1, transported.shape[-1]),
            terminals.reshape(rows, 1), rows, 1, False,
        )
        # Puffer discards advanced state for optimization (rollout=False).
        return outputs.reshape(batch, time, outputs.shape[-1]), state, OptimizationRows(inner, batch, time)

    @functools.wraps(original_backward)
    def backward(self, tape, logits, values):
        if not isinstance(tape, OptimizationRows):
            return original_backward(self, tape, logits, values)
        if logits.shape != (tape.batch, tape.time, self.output_size - 1):
            raise ValueError("Actor cotangents differ from the optimization sequence")
        if values.shape != (tape.batch, tape.time):
            raise ValueError("Critic cotangents differ from the optimization sequence")
        rows = tape.batch * tape.time
        return original_backward(
            self, tape.inner, logits.reshape(rows, 1, logits.shape[-1]), values.reshape(rows, 1),
        )

    @functools.wraps(original_critic)
    def critic(self, tape):
        return original_critic(self, tape.inner if isinstance(tape, OptimizationRows) else tape)

    NativeFabricPolicy.__init__ = initialize
    NativeFabricPolicy._forward_arrays = forward
    NativeFabricPolicy.backward_device_arrays = backward
    NativeFabricPolicy.critic_predictions = critic
    NativeFabricPolicy._generals_optimization_rows = True
    return original_forward, original_backward

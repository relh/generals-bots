"""Small memoryless Fabric policy for batched Generals observations."""

import jax
from fabric import lang as fl
from fabric import nn
from metta_training.fabric import PolicyGraph


def _silu_step(self, state, parameters, inbox, rho):
    return state.replace(pub=jax.nn.silu(inbox.drive * parameters.weight + parameters.bias))


def _silu_publish(self, state, parameters):
    return state.pub


@fl.atom
class SiLU:
    """Current-tick nonlinearity with no recurrent read or credit path."""

    visibility = fl.config(0)
    state = fl.state(pub=fl.f32(1))
    inboxes = fl.inboxes(drive=fl.slot(1, merge=fl.monoids.sum))
    params = fl.params(
        weight=fl.local((1,), init=fl.inits.constant(1.0)),
        bias=fl.local((1,), init=fl.inits.constant(0.0)),
    )
    step = _silu_step
    publish = _silu_publish


def memoryless_mlp_policy(*, observation_size: int, output_size: int, hidden: int = 64) -> PolicyGraph:
    if min(observation_size, output_size, hidden) < 1:
        raise ValueError("Policy dimensions must be positive")
    sense = nn.cluster("sense", nn.atoms.Input(), n=observation_size)
    core = nn.cluster("core", SiLU(), n=hidden)
    out = nn.cluster("out", nn.atoms.Output(), n=output_size)
    graph = nn.cluster("mlp", {"sense": sense, "core": core, "out": out})
    graph.add(
        (sense >> core).by(nn.rules.all_to_all()),
        (core >> out).by(nn.rules.all_to_all()),
    )
    return PolicyGraph(graph, "sense", ("out",))

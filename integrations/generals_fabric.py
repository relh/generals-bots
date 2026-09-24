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


def spatial_mlp_policy(
    *, observation_size: int, output_size: int, channels: int, height: int, width: int, features_per_site: int = 8
) -> PolicyGraph:
    """Memoryless local features with a global action readout."""
    if min(channels, height, width, output_size, features_per_site) < 1:
        raise ValueError("Policy dimensions must be positive")
    image_size = channels * height * width
    if observation_size < image_size:
        raise ValueError("Observation size must contain the image")
    columns, rows = (width + 1) // 2, (height + 1) // 2
    sites = columns * rows
    sense = nn.cluster(
        "sense",
        nn.atoms.Input(),
        n=observation_size,
        geometry=nn.geometry.fields(
            own={
                (f"input_{i}",): {"coord": (i % width, (i // width) % height), "image": i < image_size}
                for i in range(observation_size)
            }
        ),
    )
    core = nn.cluster(
        "core",
        SiLU(),
        n=sites * features_per_site,
        geometry=nn.geometry.fields(
            own={
                (f"a_{i}",): {"coord": (2 * (i % columns), 2 * ((i // columns) % rows))}
                for i in range(sites * features_per_site)
            }
        ),
    )
    out = nn.cluster("out", nn.atoms.Output(), n=output_size)
    graph = nn.cluster("spatial_mlp", {"sense": sense, "core": core, "out": out})
    graph.add(
        (sense >> core).by(nn.rules.stencil(src=nn.select.input_atoms().where(image=True), radius=2**0.5)),
        (core >> out).by(nn.rules.all_to_all()),
    )
    if observation_size > image_size:
        graph.add((sense >> out).by(nn.rules.all_to_all(src=nn.select.input_atoms().where(image=False))))
    return PolicyGraph(graph, "sense", ("out",))

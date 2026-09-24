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


@fl.atom
class GlobalSiLU:
    """Separate current-tick population for the global projection."""

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


def _local_kernel_key(source, target, context):
    if source[0] != "sense" or target[0] != "core":
        return ("unique", source, target)
    src_x, src_y = context.src.attr(source, "coord")
    dst_x, dst_y = context.dst.attr(target, "coord")
    return (
        context.src.attr(source, "channel"),
        dst_x - src_x,
        dst_y - src_y,
        context.dst.attr(target, "feature"),
    )


def tied_spatial_mlp_policy(
    *, observation_size: int, output_size: int, channels: int, height: int, width: int, features_per_site: int = 8
) -> PolicyGraph:
    """Local Fabric features with shared spatial kernels and feature parameters."""
    if min(channels, height, width, output_size, features_per_site) < 1:
        raise ValueError("Policy dimensions must be positive")
    image_size = channels * height * width
    if observation_size != image_size:
        raise ValueError("Tied spatial policy expects channel-first image observations")
    columns, rows = (width + 1) // 2, (height + 1) // 2
    sense = nn.cluster(
        "sense",
        nn.atoms.Input(),
        n=observation_size,
        geometry=nn.geometry.fields(
            own={
                (f"input_{i}",): {"coord": (i % width, (i // width) % height), "channel": i // (height * width)}
                for i in range(observation_size)
            }
        ),
    )
    core = nn.cluster(
        "core",
        SiLU(),
        n=columns * rows * features_per_site,
        geometry=nn.geometry.fields(
            own={
                (f"a_{i}",): {
                    "coord": (2 * ((i // features_per_site) % columns), 2 * ((i // features_per_site) // columns)),
                    "feature": i % features_per_site,
                    "core": True,
                }
                for i in range(columns * rows * features_per_site)
            }
        ),
    )
    out = nn.cluster("out", nn.atoms.Output(), n=output_size)
    graph = nn.cluster("tied_spatial_mlp", {"sense": sense, "core": core, "out": out})
    graph.add(
        (sense >> core).by(nn.rules.stencil(radius=2**0.5)).semantics(
            nn.couplings.ScalarWeighted(weight_init=fl.inits.normal(0.05))
        ),
        (core >> out).by(nn.rules.all_to_all()),
        nn.tie(core).by(nn.sharing.field("feature")).on("weight", "bias"),
        nn.tie(graph).by(nn.sharing.edge_key(_local_kernel_key)),
    )
    return PolicyGraph(graph, "sense", ("out",))


def _local_action_key(source, target, context):
    if source[0] == "sense" and target[0] == "local":
        sx, sy = context.src.attr(source, "coord")
        dx, dy = context.dst.attr(target, "coord")
        return ("input", context.src.attr(source, "channel"), dx - sx, dy - sy,
                context.dst.attr(target, "feature"))
    if source[0] == "local" and target[0] == "out":
        return ("action", context.src.attr(source, "feature"), context.dst.attr(target, "direction"))
    return ("unique", source, target)


def tied_local_action_policy(
    *, observation_size: int, output_size: int, channels: int, height: int, width: int,
    features_per_site: int = 8, global_features: int = 32, input_radius: float = 2**0.5,
) -> PolicyGraph:
    """Share board kernels and action-direction readouts, with global context."""
    cells = height * width
    if min(channels, height, width, features_per_site, global_features) < 1:
        raise ValueError("Policy dimensions must be positive")
    if observation_size != channels * cells or output_size != 4 * cells + 4:
        raise ValueError("This policy requires channel-first maps and factorized Generals actions")
    sense = nn.cluster(
        "sense", nn.atoms.Input(), n=observation_size,
        geometry=nn.geometry.fields(own={
            (f"input_{i}",): {"coord": (i % width, (i // width) % height), "channel": i // cells}
            for i in range(observation_size)
        }),
    )
    local = nn.cluster(
        "local", SiLU(), n=cells * features_per_site,
        geometry=nn.geometry.fields(own={
            (f"a_{i}",): {
                "coord": ((i // features_per_site) % width, (i // features_per_site) // width),
                "feature": i % features_per_site,
            }
            for i in range(cells * features_per_site)
        }),
    )
    global_core = nn.cluster("global", GlobalSiLU(), n=global_features)
    out = nn.cluster(
        "out", nn.atoms.Output(), n=output_size,
        geometry=nn.geometry.fields(own={
            (f"a_{i}",): {
                "coord": ((i % cells) % width, (i % cells) // width) if i < 4 * cells else (-100, -100),
                "direction": i // cells if i < 4 * cells else -1,
                "move": i < 4 * cells,
            }
            for i in range(output_size)
        }),
    )
    graph = nn.cluster("tied_local_action", {"sense": sense, "local": local, "global": global_core, "out": out})
    fixed = nn.couplings.ScalarWeighted(weight_init=fl.inits.normal(0.05))
    graph.add(
        (sense >> local).by(nn.rules.stencil(radius=input_radius)).semantics(fixed),
        (local >> out).by(nn.rules.stencil(radius=0.1)).semantics(fixed),
        (local >> global_core).by(nn.rules.all_to_all()),
        (global_core >> out).by(nn.rules.all_to_all()),
        nn.tie(local).by(nn.sharing.field("feature")).on("weight", "bias"),
        nn.tie(graph).by(nn.sharing.edge_key(_local_action_key)),
    )
    return PolicyGraph(graph, "sense", ("out",))

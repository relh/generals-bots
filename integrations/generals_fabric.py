"""Canonical memoryless 16-plane, flat-action Classic Fabric policy."""

import jax
import numpy as np
from fabric import lang as fl
from fabric import nn
from metta_training.fabric import PolicyGraph


def _silu_step(self, state, parameters, inbox, rho):
    return state.replace(pub=jax.nn.silu(inbox.drive * parameters.weight + parameters.bias))


def _silu_publish(self, state, parameters):
    return state.pub


def _product_step(self, state, parameters, inbox, rho):
    return state.replace(pub=128.0 * inbox.local * inbox.global_)


@fl.atom
class SourceGlobalProduct:
    """Multiply a site's projected context by the projected board context."""

    visibility = fl.config(0)
    state = fl.state(pub=fl.f32(1))
    inboxes = fl.inboxes(
        local=fl.slot(1, merge=fl.monoids.sum),
        global_=fl.slot(1, merge=fl.monoids.sum),
    )
    step = _product_step
    publish = _silu_publish


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


@fl.atom
class ContextSiLU:
    """Second spatial stage; keep its pool separate from the first stage."""

    visibility = fl.config(0)
    state = fl.state(pub=fl.f32(1))
    inboxes = fl.inboxes(drive=fl.slot(1, merge=fl.monoids.sum))
    params = fl.params(
        weight=fl.local((1,), init=fl.inits.constant(1.0)),
        bias=fl.local((1,), init=fl.inits.constant(0.0)),
    )
    step = _silu_step
    publish = _silu_publish


def two_stage_tied_local_action_policy(
    *, observation_size: int, output_size: int, channels: int, height: int, width: int,
    features_per_site: int = 32, global_features: int = 32, context_radius: float = 1.01,
    route_prior_strength: float = 0.5, source_army_prior_strength: float = 0.25,
    half_prior_scale: float = 0.99, full_split_prior_strength: float = 0.125,
) -> PolicyGraph:
    """Preserve the qualified topology and exact shared parameter groups."""
    if (observation_size, output_size, channels, height, width, features_per_site, global_features) != (
            7056, 3530, 16, 21, 21, 32, 32):
        raise ValueError("Classic Fabric requires the canonical 16-plane F32/G32 flat model")
    if context_radius not in (1.01, 2.01):
        raise ValueError("Classic context radius must be 1.01 or 2.01")
    if not all(np.isfinite(value) and value > 0 for value in (
            route_prior_strength, source_army_prior_strength, full_split_prior_strength)):
        raise ValueError("Current public action priors must be finite and positive")
    if not np.isfinite(half_prior_scale) or not 0 <= half_prior_scale <= 1:
        raise ValueError("Half-army prior scale must be in [0, 1]")
    cells, move_logits = 441, 3528
    sense = nn.cluster(
        "sense", nn.atoms.Input(), n=observation_size,
        geometry=nn.geometry.fields(own={
            (f"input_{i}",): {"coord": (i % width, (i // width) % height), "channel": i // cells}
            for i in range(observation_size)
        }),
    )

    def site_layer(name, atom):
        return nn.cluster(
            name, atom, n=cells * features_per_site,
            geometry=nn.geometry.fields(own={
                (f"a_{i}",): {
                    "coord": ((i // features_per_site) % width, (i // features_per_site) // width),
                    "feature": i % features_per_site,
                }
                for i in range(cells * features_per_site)
            }),
        )

    local = site_layer("local", SiLU())
    context = site_layer("context", ContextSiLU())
    global_core = nn.cluster("global", GlobalSiLU(), n=global_features)
    product = nn.cluster(
        "product", SourceGlobalProduct(), n=cells * 8,
        geometry=nn.geometry.fields(own={
            (f"a_{i}",): {"coord": ((i // 8) % width, (i // 8) // width), "rank": i % 8}
            for i in range(cells * 8)
        }),
    )
    out = nn.cluster(
        "out", nn.atoms.Output(), n=output_size,
        geometry=nn.geometry.fields(own={
            (f"a_{i}",): {
                "coord": ((i % cells) % width, (i % cells) // width) if i < move_logits else (-100, -100),
                "direction": (i // cells) % 4 if i < move_logits else -1,
                "move": i < move_logits,
                "readout_group": i // cells if i < move_logits else i,
                **({"split": i // (4 * cells)} if i < move_logits else {}),
            }
            for i in range(output_size)
        }),
    )
    graph = nn.cluster("two_stage_tied_local_action", {
        "sense": sense, "local": local, "context": context, "global": global_core,
        "product": product, "out": out,
    })
    fixed = nn.couplings.ScalarWeighted(weight_init=fl.inits.normal(0.05))

    def edge_key(source, target, geometry):
        src_kind, dst_kind = source[0], target[0]
        if (src_kind, dst_kind) == ("sense", "local"):
            sx, sy = geometry.src.attr(source, "coord")
            dx, dy = geometry.dst.attr(target, "coord")
            return ("input", geometry.src.attr(source, "channel"), dx - sx, dy - sy,
                    geometry.dst.attr(target, "feature"))
        if (src_kind, dst_kind) == ("local", "context"):
            sx, sy = geometry.src.attr(source, "coord")
            dx, dy = geometry.dst.attr(target, "coord")
            return ("context", geometry.src.attr(source, "feature"), dx - sx, dy - sy,
                    geometry.dst.attr(target, "feature"))
        if (src_kind, dst_kind) == ("context", "out"):
            return ("action", geometry.src.attr(source, "feature"), geometry.dst.attr(target, "direction"),
                    geometry.dst.attr(target, "split"))
        if (src_kind, dst_kind) == ("global", "out"):
            return ("global", source[1], geometry.dst.attr(target, "readout_group"))
        if (src_kind, dst_kind) == ("context", "product"):
            return ("product_local", geometry.src.attr(source, "feature"), geometry.dst.attr(target, "rank"))
        if (src_kind, dst_kind) == ("global", "product"):
            return ("product_global", source[1], geometry.dst.attr(target, "rank"))
        if (src_kind, dst_kind) == ("product", "out"):
            return ("product_out", geometry.src.attr(source, "rank"),
                    geometry.dst.attr(target, "direction"), geometry.dst.attr(target, "split"))
        if (src_kind, dst_kind) == ("sense", "out"):
            channel = geometry.src.attr(source, "channel")
            if channel == 0 and source_army_prior_strength:
                return ("source_army", geometry.dst.attr(target, "split"))
            if channel == 4 and full_split_prior_strength:
                return ("full_split",)
            if route_prior_strength:
                return ("route", geometry.dst.attr(target, "direction"), geometry.dst.attr(target, "split"))
        return ("unique", source, target)

    graph.add(
        (sense >> local).by(nn.rules.stencil(radius=0.1)).semantics(fixed),
        (local >> context).by(nn.rules.stencil(radius=context_radius)).semantics(fixed),
        (context >> out).by(nn.rules.stencil(radius=0.1)).semantics(fixed),
        (context >> global_core).by(nn.rules.all_to_all()),
        (global_core >> out).by(nn.rules.all_to_all()),
        (context >> product).by(nn.rules.stencil(radius=0.1)).semantics(fixed).into_("local"),
        (global_core >> product).by(nn.rules.all_to_all()).semantics(fixed).into_("global_"),
        (product >> out).by(nn.rules.stencil(radius=0.1)).semantics(
            nn.couplings.ScalarWeighted(weight_init=fl.inits.constant(0.0))
        ),
        nn.tie(local).by(nn.sharing.field("feature")).on("weight", "bias"),
        nn.tie(context).by(nn.sharing.field("feature")).on("weight", "bias"),
        nn.tie(graph).by(nn.sharing.edge_key(edge_key)),
        nn.tie(out).by(nn.sharing.field("readout_group")).on("W", "b"),
    )
    for split in range(2):
        graph.add((sense >> out).by(nn.rules.edges(np.asarray(
            [((7 + direction) * cells + cell, (4 * split + direction) * cells + cell)
             for direction in range(4) for cell in range(cells)], dtype=np.int32
        ))).semantics(nn.couplings.ScalarWeighted(
            weight_init=fl.inits.constant(route_prior_strength * (half_prior_scale if split else 1.0))
        )))
    for split in range(2):
        graph.add((sense >> out).by(nn.rules.edges(np.asarray(
            [(cell, (4 * split + direction) * cells + cell)
             for direction in range(4) for cell in range(cells)], dtype=np.int32
        ))).semantics(nn.couplings.ScalarWeighted(
            weight_init=fl.inits.constant(source_army_prior_strength * (half_prior_scale if split else 1.0))
        )))
    graph.add((sense >> out).by(nn.rules.edges(np.asarray(
        [(4 * cells + cell, direction * cells + cell)
         for direction in range(4) for cell in range(cells)], dtype=np.int32
    ))).semantics(nn.couplings.ScalarWeighted(weight_init=fl.inits.constant(full_split_prior_strength))))
    return PolicyGraph(graph, "sense", ("out",))

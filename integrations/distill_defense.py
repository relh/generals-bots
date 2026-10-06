"""Bounded GPU warm start from independent public Sentinel defense examples.

Supervised updates never increment the reinforcement-learning environment clock.
Native flat parameters are authoritative, preserving Fabric sharing and stencil masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from generals.agents.sentinel_agent import SentinelAgent
from integrations.classic_contract import verify_engine
from integrations.classic_position_curriculum import load_positions
from integrations.generate_defense_curriculum import OFFSETS, audit_position
from integrations.spatial_action_sampling import (
    acting_logits,
    public_doomed_attack_route_penalty,
    public_early_route_temperature,
    public_neutral_route_bonus,
    public_weak_owned_route_penalty,
)
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


def adam_step(parameters, gradient, state, learning_rate):
    """Canonical Adam on one flat vector, after global L2 clipping at one."""
    first, second, count = state
    norm = jnp.linalg.norm(gradient)
    clipped = gradient * jnp.minimum(1.0, 1.0 / jnp.maximum(norm, 1e-12))
    first = 0.9 * first + 0.1 * clipped
    second = 0.999 * second + 0.001 * clipped * clipped
    count = count + 1
    corrected_first = first / (1 - 0.9**count)
    corrected_second = second / (1 - 0.999**count)
    after = parameters - learning_rate * corrected_first / (jnp.sqrt(corrected_second) + 1e-8)
    return after, (first, second, count), jnp.isfinite(norm)


def native_layout(model):
    """The inverse of export_spatial_policy_bundle's realized graph gather."""
    names = (
        "input_kernel",
        "context_kernel",
        "action_kernel",
        "global_weight",
        "global_bias",
        "global_kernel",
        "readout_kernel",
        "output_weight",
        "output_bias",
    )
    indices = {name: np.asarray(getattr(model, name)) for name in names}
    for name in ("local_weight", "local_bias", "context_weight", "context_bias"):
        indices[name] = np.asarray(getattr(model, name)[0])
    constants = {}
    for i, (source, lookup) in enumerate(model.priors):
        constants[f"prior_source_{i}"] = np.asarray(source)
        indices[f"prior_weight_{i}"] = np.asarray(lookup)
    return indices, constants


def gather_weights(parameters, indices, constants, xp=jnp):
    # Negative entries are absent stencil/connection weights, never trainable.
    weights = {
        name: xp.where(xp.asarray(index) >= 0, parameters[xp.maximum(xp.asarray(index), 0)], 0)
        for name, index in indices.items()
    }
    return dict(weights, **{name: xp.asarray(value) for name, value in constants.items()})


def serving_logits(policy, outputs, values, masks):
    temperature = policy.move_temperature
    if policy.early_route_temperature is not None:
        temperature = public_early_route_temperature(
            values, temperature, policy.early_route_temperature, policy.early_route_turns, jnp
        )
    logits = acting_logits(
        outputs, temperature, policy.split_temperature, jnp, route_half_weight=policy.route_half_weight
    )[:, :3529]
    for coefficient, transform in (
        (policy.neutral_route_bias, public_neutral_route_bonus),
        (policy.weak_owned_route_penalty, public_weak_owned_route_penalty),
        (policy.doomed_attack_route_penalty, public_doomed_attack_route_penalty),
    ):
        if coefficient:
            logits += transform(values, coefficient, jnp)
    logits /= policy.full_action_temperature
    if policy.log_gap_scale:
        from integrations.spatial_exploration import log_gap_logits

        logits = log_gap_logits(logits, masks, policy.log_gap_scale, jnp)
    if policy.capital_safety:
        from integrations.capital_safety import constrain_logits

        logits = constrain_logits(logits, values, jnp)
    return jnp.where(masks, logits, -jnp.inf)


def cross_entropy(logits, targets):
    return -jax.nn.log_softmax(logits)[jnp.arange(len(targets)), targets].mean()


def dataset(manifest_path):
    """Teacher sees public observations only; synthetic oracle is diagnostic."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "classic-midgame-positions-v1"
        or manifest.get("engine_sha256") != verify_engine()
        or "root_training_seed" not in manifest
        or manifest.get("teacher_labels_enabled") is not False
    ):
        raise ValueError("Require freshly generated train-only synthetic curriculum")
    positions = load_positions(manifest_path.parent / "positions.npz", manifest["positions_sha256"])
    rows = manifest["provenance"]
    if len(rows) != len(positions.time):
        raise ValueError("Position provenance count differs")
    values, masks, labels, defenses = [], [], [], []
    teacher = SentinelAgent(build_castles=False, deathtouch_turn=None, max_turns=2000)
    from integrations.puffer_codec import encode_coworld_directional_observation

    for i, row in enumerate(rows):
        state = jax.tree.map(lambda x: x[i], positions)
        seat = row["seat"]
        source_grid = np.where(state.mountains, -2, np.where(state.castles, state.armies, 0)).astype(np.int32)
        original_generals = np.asarray(state.general_positions)[:: -1 if seat else 1]
        for side, position in enumerate(original_generals):
            source_grid[tuple(position)] = side + 1
        if hashlib.sha256(source_grid.tobytes()).hexdigest() != row["map_sha256"]:
            raise ValueError("Actual position terrain/general source differs from its recorded map identity")
        capital = np.asarray(state.general_positions[seat])
        attacker = capital + OFFSETS[row["attack_direction"]]
        attack = jnp.array([0, *attacker, (1, 0, 3, 2)[row["attack_direction"]], 0], jnp.int32)
        obs, indices, actions, survivors, audit = audit_position(state, attack, seat)
        if not audit["defendable"] or not audit["pass_loses_capital"]:
            raise ValueError("Synthetic exercise is not a credible defendable threat")
        action = np.asarray(teacher.act(obs, jax.random.PRNGKey(row["map_seed"])))
        matches = np.all(np.asarray(actions) == action, axis=1)
        if not matches.any() or not survivors[matches].any():
            raise ValueError("Public Sentinel teacher must demonstrate a legal surviving action for every example")
        encoded, mask = encode_coworld_directional_observation(obs)
        label = int((action[4] * 4 + action[3]) * 441 + action[1] * 21 + action[2])
        if action[0]:
            label = 3528
        if not bool(mask[label]):
            raise ValueError("Teacher label is outside the public legal mask")
        safe = np.zeros(3529, bool)
        safe[indices[survivors]] = True
        values.append(np.asarray(encoded))
        masks.append(np.asarray(mask))
        labels.append(label)
        defenses.append(safe)
    return {
        "values": np.asarray(values, np.float32),
        "masks": np.asarray(masks, bool),
        "targets": np.asarray(labels, np.int32),
        "defenses": np.asarray(defenses, bool),
        "map_hashes": {row["map_sha256"] for row in rows},
        "map_seeds": {row["map_seed"] for row in rows},
        "root_seed": manifest["root_training_seed"],
        "sha256": manifest["positions_sha256"],
    }


def prepare_encoded(manifest_path, output):
    """Seal CPU-prepared public teacher labels so GPU jobs only load arrays."""
    output = Path(output)
    if output.exists():
        raise ValueError("Encoded dataset output must be a new directory")
    data = dataset(manifest_path)
    output.mkdir(parents=True)
    archive = output / "examples.npz"
    np.savez_compressed(archive, **{name: data[name] for name in ("values", "masks", "targets", "defenses")})
    import inspect

    metadata = {
        "schema": "generals-defense-public-labels-v1",
        "engine_sha256": verify_engine(),
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "positions_sha256": data["sha256"],
        "source_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        "teacher_source_sha256": hashlib.sha256(Path(inspect.getfile(SentinelAgent)).read_bytes()).hexdigest(),
        "root_seed": data["root_seed"],
        "map_hashes": sorted(data["map_hashes"]),
        "map_seeds": sorted(data["map_seeds"]),
        "examples": len(data["targets"]),
        "scope": "Public Sentinel labels from independent synthetic training maps; no replay inputs or CPU training.",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_encoded(path, manifest=None):
    """Load the canonical sealed public-label format; GPU training never prepares labels."""
    import inspect

    if manifest is None:
        manifest = json.loads(Path(path).read_text())
    if manifest.get("schema") != "generals-defense-public-labels-v1":
        raise ValueError("Distillation requires sealed pre-encoded public teacher data")

    archive = Path(path).parent / "examples.npz"
    if (
        manifest["engine_sha256"] != verify_engine()
        or manifest["teacher_source_sha256"]
        != hashlib.sha256(Path(inspect.getfile(SentinelAgent)).read_bytes()).hexdigest()
        or archive.stat().st_size > 64 * 1024**2
        or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]
    ):
        raise ValueError("Encoded public teacher dataset identity differs")
    with np.load(archive, allow_pickle=False) as saved:
        if set(saved.files) != {"values", "masks", "targets", "defenses"}:
            raise ValueError("Encoded teacher array fields differ")
        arrays = {name: saved[name] for name in saved.files}
    count = manifest["examples"]
    shapes = {"values": (count, 7056), "masks": (count, 3529), "targets": (count,), "defenses": (count, 3529)}
    dtypes = {"values": np.float32, "targets": np.int32, "masks": bool, "defenses": bool}
    if not 1 <= count <= 4096 or any(arrays[k].shape != shapes[k] or arrays[k].dtype != dtypes[k] for k in shapes):
        raise ValueError("Encoded teacher array shapes or dtypes differ")
    targets = arrays["targets"]
    if (
        not np.isfinite(arrays["values"]).all()
        or np.any((targets < 0) | (targets >= 3529))
        or not arrays["masks"][np.arange(count), targets].all()
        or not arrays["defenses"][np.arange(count), targets].all()
        or np.any(arrays["defenses"] & ~arrays["masks"])
    ):
        raise ValueError("Encoded labels must be finite, legal public Sentinel defenses")
    return dict(
        arrays,
        root_seed=manifest["root_seed"],
        map_hashes=set(manifest["map_hashes"]),
        map_seeds=set(manifest["map_seeds"]),
        sha256=manifest["sha256"],
    )


def validate_split(train, heldout):
    if (
        train["root_seed"] == heldout["root_seed"]
        or train["map_hashes"] & heldout["map_hashes"]
        or train["map_seeds"] & heldout["map_seeds"]
    ):
        raise ValueError("Training and held-out synthetic maps must be independent")
    if min(len(train["map_hashes"]), len(heldout["map_hashes"])) < 16:
        raise ValueError("Require at least 16 independent maps in each split")


def write_checkpoint(path, parameters, words):
    parameters = np.asarray(parameters)
    if parameters.dtype != np.float32 or parameters.shape != (words,) or not np.isfinite(parameters).all():
        raise ValueError("Native checkpoint must contain the exact finite float32 parameter vector")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(parameters.astype("<f4", copy=False).tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    bundle,
    train_manifest,
    heldout_manifest,
    factory_source,
    output,
    *,
    updates=256,
    batch_size=128,
    learning_rate=1e-4,
    seed=7600101,
):
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Defense distillation requires GPU execution; CPU training is forbidden")
    if not 1 <= updates <= 1024 or not 1 <= batch_size <= 512 or not 0 < learning_rate <= 1e-3:
        raise ValueError("Require bounded updates<=1024, batch<=512, learning_rate<=0.001")
    output, bundle, factory_source = Path(output), Path(bundle), Path(factory_source)
    if output.exists():
        raise ValueError("Distillation output must be a new directory")
    policy = SpatialPlayerPolicy(bundle)
    if policy.channels != 16 or policy.action_mode != "structured_sample":
        raise ValueError("Require the current sampled public-scalar spatial policy")
    from integrations.native_spatial_asset import canonical_json, write_asset
    from integrations.export_spatial_policy_bundle import realized_model

    asset = policy.asset
    source_sha = hashlib.sha256(factory_source.read_bytes()).hexdigest()
    native, model, model_sha, abi_sha = realized_model(
        canonical_json(asset.metadata["fabric"]).decode(), factory_source, source_sha)
    asset.verify_target(factory_source_sha256=source_sha, model_sha256=model_sha, abi_sha256=abi_sha)
    ancestor_seeds = set(asset.metadata["training_seeds"])
    training, heldout = load_encoded(train_manifest), load_encoded(heldout_manifest)
    validate_split(training, heldout)
    if (heldout["map_seeds"] | {heldout["root_seed"]}) & (ancestor_seeds | {seed}):
        raise ValueError("Held-out teacher views overlap the source training lineage")
    indices, constants = native_layout(model)
    original = np.frombuffer(asset.policy, "<f4").copy()
    if original.size != native.buffers.parameter_words:
        raise ValueError("Native checkpoint word count differs")
    recovered = gather_weights(original, indices, constants, np)
    for name, weight in policy.weights.items():
        if not np.array_equal(weight, recovered[name]):
            raise ValueError("Native gather differs from exported portable weights: " + name)
    parameters = jnp.asarray(original)
    optimizer_state = (jnp.zeros_like(parameters), jnp.zeros_like(parameters), jnp.int32(0))

    def logits(parameters, values, masks):
        weights = gather_weights(parameters, indices, constants)
        return serving_logits(policy, policy._forward(values, jnp, weights=weights), values, masks)

    @jax.jit
    def update(parameters, optimizer_state, values, masks, targets):
        loss, gradient = jax.value_and_grad(lambda p: cross_entropy(logits(p, values, masks), targets))(parameters)
        after, optimizer_state, finite_norm = adam_step(parameters, gradient, optimizer_state, learning_rate)
        finite = finite_norm & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(after))
        return after, optimizer_state, loss, finite

    @jax.jit
    def metrics(parameters, values, masks, targets, defenses):
        prediction = logits(parameters, values, masks)
        return jnp.array(
            [
                cross_entropy(prediction, targets),
                jnp.mean(jnp.argmax(prediction, axis=1) == targets),
                jnp.mean(jnp.sum(jax.nn.softmax(prediction) * defenses, axis=1)),
            ]
        )

    def measure(data):
        scores = []
        for start in range(0, len(data["targets"]), batch_size):
            rows = slice(start, start + batch_size)
            scores.append(
                (
                    len(data["targets"][rows]),
                    np.asarray(
                        metrics(
                            parameters,
                            *[jnp.asarray(data[name][rows]) for name in ("values", "masks", "targets", "defenses")],
                        )
                    ),
                )
            )
        return dict(
            zip(
                ("cross_entropy", "teacher_top1", "defense_survival_probability"),
                np.average([score for _, score in scores], axis=0, weights=[count for count, _ in scores]).tolist(),
            )
        )

    device_training = {name: jax.device_put(training[name]) for name in ("values", "masks", "targets")}
    before = {"train": measure(training), "heldout": measure(heldout)}
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    losses = []
    for iteration in range(updates):
        rows = rng.integers(0, len(training["targets"]), batch_size)
        parameters, optimizer_state, loss, finite = update(
            parameters, optimizer_state, *[device_training[name][rows] for name in ("values", "masks", "targets")]
        )
        if not bool(finite) or not np.isfinite(float(loss)):
            raise FloatingPointError("Distillation produced a nonfinite update")
        losses.append(float(loss))
        if iteration == 0 or (iteration + 1) % 32 == 0 or iteration + 1 == updates:
            print(
                json.dumps(
                    {
                        "supervised_update": iteration + 1,
                        "cross_entropy": losses[-1],
                        "examples": (iteration + 1) * batch_size,
                        "wall_seconds": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    elapsed = time.monotonic() - start
    after = {"train": measure(training), "heldout": measure(heldout)}
    output.mkdir(parents=True)
    checkpoint = output / "supervised.bin"
    digest = write_checkpoint(checkpoint, parameters, native.buffers.parameter_words)
    provenance = dict(operation="supervised", ancestors={
        "source_asset": hashlib.sha256((bundle / "asset.json").read_bytes()).hexdigest(),
        "source_policy": asset.metadata["policy_sha256"],
        "training_data": training["sha256"],
    }, reinforcement_learning_steps_added=0, optimizer_updates=updates,
        optimizer_seed=seed, optimizer_learning_rate=learning_rate,
        heldout_data_sha256=heldout["sha256"],
        training_map_seeds=sorted(training["map_seeds"]), heldout_map_seeds=sorted(heldout["map_seeds"]))
    asset_path = write_asset(output / "asset", fabric=asset.metadata["fabric"],
        factory_source_sha256=source_sha, model_sha256=model_sha, abi_sha256=abi_sha,
        policy=checkpoint, sampler=asset.metadata["sampler"], provenance=provenance,
        training_seeds=sorted(ancestor_seeds | training["map_seeds"] | {seed, training["root_seed"]}))
    from integrations.export_spatial_policy_bundle import export_bundle

    export_bundle(asset_path, hashlib.sha256(asset_path.read_bytes()).hexdigest(), factory_source, output / "bundle")
    exported = SpatialPlayerPolicy(output / "bundle")
    for name, value in gather_weights(np.asarray(parameters), indices, constants, np).items():
        if not np.array_equal(exported.weights[name], value):
            raise ValueError("Learned checkpoint re-export differs: " + name)
    report = {
        "before": before,
        "after": after,
        "updates": updates,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "wall_seconds": elapsed,
        "examples_per_second": updates * batch_size / elapsed,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "gpu": str(jax.devices()[0]),
        "gpu_model": jax.devices()[0].device_kind,
        "independent_training_maps": len(training["map_hashes"]),
        "independent_heldout_maps": len(heldout["map_hashes"]),
        "supervised_examples": updates * batch_size,
        "reinforcement_learning_steps_added": 0,
        "checkpoint_sha256": digest,
        "scope": "Supervised GPU warm start; no PPO SPS or hosted-strength claim.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "train-manifest", "heldout-manifest", "factory-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7600101)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.bundle,
                args.train_manifest,
                args.heldout_manifest,
                args.factory_source,
                args.output,
                updates=args.updates,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Fit a frozen opponent from hosted replays without teaching the RL learner.

This is an opponent-pool experiment, not a learner loss. A seed-disjoint
validation set screens behavior cloning before any RL run uses the result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.spatial_action_sampling import (
    acting_logits, public_doomed_attack_route_penalty,
    public_neutral_route_bonus, public_weak_owned_route_penalty,
)
from integrations.spatial_policy_bundle import SpatialPlayerPolicy


# The Fabric output bias has tied indices, so fitting its 3,530 portable
# positions independently cannot round-trip into a native checkpoint.
TRAINABLE_NAMES = ("action_kernel", "readout_kernel")


def load_dataset(root: Path, max_shards: int | None):
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "coworld-classic-opponent-replay-v1":
        raise ValueError("Expected verified Classic opponent replay dataset")
    entries = manifest["shards"][:max_shards]
    if not entries:
        raise ValueError("No replay shards selected")
    selected = {part: {name: [] for name in ("observations", "masks", "actions", "games")}
                for part in ("train", "validation")}
    game_offset = 0
    for entry in entries:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Replay shard checksum differs: {path}")
        with np.load(path) as shard:
            observations = shard["observations"]
            masks = np.unpackbits(shard["mask_bits"], axis=1, bitorder="little")[:, :3529].astype(bool)
            actions = shard["actions"].astype(np.int32)
            games = shard["game_indices"].astype(np.int32)
            validation = shard["validation_games"]
            if (observations.shape != (len(actions), 7056) or masks.shape != (len(actions), 3529)
                    or len(validation) != entry["games"] or len(actions) != entry["actions"]
                    or not np.all(masks[np.arange(len(actions)), actions])):
                raise ValueError(f"Invalid replay shard contract: {path}")
            for part, chosen in (("train", ~validation[games]), ("validation", validation[games])):
                target = selected[part]
                target["observations"].append(observations[chosen])
                target["masks"].append(masks[chosen])
                target["actions"].append(actions[chosen])
                target["games"].append(games[chosen] + game_offset)
            game_offset += len(validation)
    result = {part: {name: np.concatenate(parts) for name, parts in data.items()}
              for part, data in selected.items()}
    if min(len(result[part]["actions"]) for part in result) == 0:
        raise ValueError("Both training and validation games are required")
    return manifest, result


def fit(bundle: Path, dataset: Path, output: Path, *, steps: int, batch_size: int,
        learning_rate: float, route_temperature: float, seed: int, max_shards: int | None,
        validation_interval: int, mode: str = "full_action"):
    if output.exists():
        raise FileExistsError(output)
    if (steps < 1 or batch_size < 1 or validation_interval < 1 or not np.isfinite(learning_rate)
            or learning_rate <= 0 or not np.isfinite(route_temperature) or route_temperature <= 0):
        raise ValueError("Require positive finite training settings")
    manifest, dataset_parts = load_dataset(dataset, max_shards)
    if mode not in ("full_action", "conditional_split"):
        raise ValueError("Unknown replay opponent fit mode")
    if mode == "conditional_split":
        for part, data in dataset_parts.items():
            chosen = data["actions"] < 3528
            dataset_parts[part] = {name: values[chosen] for name, values in data.items()}
            if not chosen.any():
                raise ValueError(f"No nonpass actions in {part}")
    policy = SpatialPlayerPolicy(bundle)
    if policy.action_mode != "structured_sample" or policy.channels != 16:
        raise ValueError("Replay opponent requires the structured sixteen-plane Classic policy")
    if mode == "conditional_split" and route_temperature != policy.move_temperature:
        raise ValueError("Conditional split fit must preserve the source route temperature")
    frozen = {name: jnp.asarray(value) for name, value in policy.weights.items() if name not in TRAINABLE_NAMES}
    initial = {name: jnp.asarray(policy.weights[name]) for name in TRAINABLE_NAMES}
    trainable_mask = {
        "action_kernel": jnp.asarray(np.broadcast_to(np.arange(8) >= 4, initial["action_kernel"].shape)),
        "readout_kernel": jnp.asarray(np.broadcast_to(
            (np.arange(3530) >= 1764) & (np.arange(3530) < 3528), initial["readout_kernel"].shape)),
    }
    params = initial
    moments = jax.tree.map(jnp.zeros_like, params)
    variances = jax.tree.map(jnp.zeros_like, params)
    device_data = {part: {name: jax.device_put(value) for name, value in values.items() if name != "games"}
                   for part, values in dataset_parts.items()}

    def action_logits(weights, observations, masks):
        all_weights = {**frozen, **weights}
        values = observations.astype(jnp.float32)
        outputs = policy._forward(values, jnp, all_weights)
        logits = acting_logits(outputs, route_temperature, policy.split_temperature, jnp)[:, :3529]
        if policy.neutral_route_bias:
            logits += public_neutral_route_bonus(values, policy.neutral_route_bias, jnp)
        if policy.weak_owned_route_penalty:
            logits += public_weak_owned_route_penalty(values, policy.weak_owned_route_penalty, jnp)
        if policy.doomed_attack_route_penalty:
            logits += public_doomed_attack_route_penalty(values, policy.doomed_attack_route_penalty, jnp)
        return jnp.where(masks, logits, -1e9)

    def cross_entropy(weights, observations, masks, actions):
        if mode == "conditional_split":
            outputs = policy._forward(observations.astype(jnp.float32), jnp, {**frozen, **weights})
            routes = actions % 1764
            full = jnp.take_along_axis(outputs[:, :1764], routes[:, None], axis=1)[:, 0]
            half = jnp.take_along_axis(outputs[:, 1764:3528], routes[:, None], axis=1)[:, 0]
            difference = (half - full) / policy.split_temperature
            is_half = actions >= 1764
            return jax.nn.softplus(difference) - is_half * difference
        log_prob = jax.nn.log_softmax(action_logits(weights, observations, masks))
        return -jnp.take_along_axis(log_prob, actions[:, None], axis=1)[:, 0]

    @jax.jit
    def train_step(weights, first, second, step, key, observations, masks, actions):
        rows = jax.random.randint(key, (batch_size,), 0, len(actions))

        def loss(current):
            return cross_entropy(current, observations[rows], masks[rows], actions[rows]).mean()

        value, gradient = jax.value_and_grad(loss)(weights)
        if mode == "conditional_split":
            gradient = jax.tree.map(lambda g, mask: jnp.where(mask, g, 0), gradient, trainable_mask)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in gradient.values()))
        gradient = jax.tree.map(lambda g: g * jnp.minimum(1.0, 1.0 / jnp.maximum(norm, 1e-8)), gradient)
        first = jax.tree.map(lambda old, g: .9 * old + .1 * g, first, gradient)
        second = jax.tree.map(lambda old, g: .999 * old + .001 * g * g, second, gradient)
        next_weights = jax.tree.map(
            lambda w, m, v: w - learning_rate * (m / (1 - .9**step)) /
            (jnp.sqrt(v / (1 - .999**step)) + 1e-8), weights, first, second)
        return next_weights, first, second, value, norm

    @jax.jit
    def validation_batch(weights, observations, masks, actions):
        losses = cross_entropy(weights, observations, masks, actions)
        half = (actions >= 1764) & (actions < 3528)
        if mode == "conditional_split":
            outputs = policy._forward(observations.astype(jnp.float32), jnp, {**frozen, **weights})
            routes = actions % 1764
            full = jnp.take_along_axis(outputs[:, :1764], routes[:, None], axis=1)[:, 0]
            split = jnp.take_along_axis(outputs[:, 1764:3528], routes[:, None], axis=1)[:, 0]
            predicted_half = jax.nn.sigmoid((split - full) / policy.split_temperature)
            return jnp.sum(losses), jnp.sum(losses * half), jnp.sum(half), jnp.sum(predicted_half)
        return jnp.sum(losses), jnp.sum(losses * half), jnp.sum(half), jnp.sum(half, dtype=jnp.float32)

    def validate(weights):
        data = device_data["validation"]
        totals = np.zeros(4, np.float64)
        for start in range(0, len(data["actions"]), 512):
            stop = min(start + 512, len(data["actions"]))
            totals += np.asarray(validation_batch(weights, data["observations"][start:stop],
                                                  data["masks"][start:stop], data["actions"][start:stop]))
        return {"nll": float(totals[0] / len(data["actions"])),
                "half_nll": float(totals[1] / totals[2]) if totals[2] else None,
                "half_actions": int(totals[2]), "actions": len(data["actions"]),
                "predicted_half_rate": float(totals[3] / len(data["actions"]))}

    history = [{"step": 0, "validation": validate(params)}]
    best_step = 0
    best_nll = history[0]["validation"]["nll"]
    best_params = {name: np.asarray(value).copy() for name, value in params.items()}
    print(json.dumps(history[-1]), flush=True)
    for step in range(1, steps + 1):
        key = jax.random.PRNGKey(seed + step)
        params, moments, variances, train_loss, grad_norm = train_step(
            params, moments, variances, step, key,
            device_data["train"]["observations"], device_data["train"]["masks"],
            device_data["train"]["actions"])
        if step % validation_interval == 0 or step == steps:
            row = {"step": step, "train_batch_nll": float(train_loss),
                   "gradient_norm": float(grad_norm), "validation": validate(params)}
            if not np.isfinite(row["train_batch_nll"]) or not np.isfinite(row["validation"]["nll"]):
                raise FloatingPointError("Nonfinite opponent clone loss")
            history.append(row)
            if row["validation"]["nll"] < best_nll:
                best_step = step
                best_nll = row["validation"]["nll"]
                best_params = {name: np.asarray(value).copy() for name, value in params.items()}
            print(json.dumps(row), flush=True)
    output.mkdir(parents=True)
    weights = {**policy.weights, **best_params}
    if mode == "conditional_split":
        for name in TRAINABLE_NAMES:
            weights[name] = np.where(np.asarray(trainable_mask[name]), weights[name], policy.weights[name])
            if not np.array_equal(weights[name][~np.asarray(trainable_mask[name])],
                                  policy.weights[name][~np.asarray(trainable_mask[name])]):
                raise AssertionError(f"Frozen route weights changed: {name}")
    np.savez_compressed(output / "weights.npz", **weights)
    result = {"schema": "coworld-classic-replay-opponent-fit-v1", "source_policy_sha256":
              hashlib.sha256((bundle / "policy.bin").read_bytes()).hexdigest(),
              "dataset_manifest_sha256": hashlib.sha256((dataset / "dataset.json").read_bytes()).hexdigest(),
              "dataset_opponent_policy_version_id": manifest["opponent_policy_version_id"],
              "train_games": len(np.unique(dataset_parts["train"]["games"])),
              "validation_games": len(np.unique(dataset_parts["validation"]["games"])),
              "train_actions": len(dataset_parts["train"]["actions"]),
              "validation_actions": len(dataset_parts["validation"]["actions"]),
              "steps": steps, "batch_size": batch_size, "learning_rate": learning_rate,
              "mode": mode,
              "route_temperature": route_temperature, "split_temperature": policy.split_temperature,
              "early_route_temperature": policy.early_route_temperature if mode == "conditional_split" else None,
              "early_route_turns": policy.early_route_turns if mode == "conditional_split" else None,
              "trainable_names": TRAINABLE_NAMES, "seed": seed, "history": history,
              "selected_step": best_step, "selected_validation_nll": best_nll,
              "weights_sha256": hashlib.sha256((output / "weights.npz").read_bytes()).hexdigest()}
    (output / "fit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--route-temperature", type=float, default=0.2)
    parser.add_argument("--mode", choices=("full_action", "conditional_split"), default="full_action")
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--seed", type=int, default=12091)
    args = parser.parse_args()
    fit(args.bundle, args.dataset, args.output, steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, route_temperature=args.route_temperature,
        validation_interval=args.validation_interval, max_shards=args.max_shards, seed=args.seed,
        mode=args.mode)


if __name__ == "__main__":
    main()

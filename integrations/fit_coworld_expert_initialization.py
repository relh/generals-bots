"""Bounded behavior-cloning initialization for the existing flat Fabric actor.

This writes policy weights only. Any later Puffer RL run uses a fresh optimizer
and has no expert targets or teacher loss.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig


def make_batch(dataset, rows, batch_size, input_size):
    valid = len(rows)
    obs = np.zeros((batch_size, 1, input_size), np.float32)
    obs[:valid, 0, :dataset["observations"].shape[1]] = dataset["observations"][rows]
    mask = np.zeros((batch_size, 3529), bool)
    mask[:valid] = np.unpackbits(dataset["action_masks"][rows], axis=1, count=3529).astype(bool)
    mask[valid:, -1] = True
    label = np.full((batch_size,), 3528, np.int32)
    label[:valid] = dataset["actions"][rows]
    weight = np.zeros((batch_size,), np.float32)
    weight[:valid] = 1
    return jnp.asarray(obs), jnp.asarray(mask), jnp.asarray(label), jnp.asarray(weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--anchor", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=740)
    args = parser.parse_args()
    assert args.epochs > 0 and args.batch_size > 0 and args.learning_rate > 0 and args.anchor >= 0
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    packed = np.load(args.dataset)
    metadata = json.loads(str(packed["metadata"]))
    data = {key: packed[key] for key in ("observations", "action_masks", "actions", "episode")}
    packed.close()
    assert data["observations"].shape[1] == 11 * 21 * 21
    assert data["action_masks"].shape[1] == 442
    assert len(data["actions"]) == sum(e["end"]-e["start"] for e in metadata["episodes"])
    train_games = [i for i, e in enumerate(metadata["episodes"]) if e["split"] == "train"]
    validation_games = [i for i, e in enumerate(metadata["episodes"]) if e["split"] == "holdout"]
    assert train_games and validation_games and not set(train_games) & set(validation_games)
    train_rows = np.flatnonzero(np.isin(data["episode"], train_games))
    validation_rows = np.flatnonzero(np.isin(data["episode"], validation_games))
    policy = FrozenPolicy(FrozenPolicyConfig(
        build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0",
    ))
    assert policy.policy.action_sizes == [3529]
    with jax.default_device(policy.device):
        initial = jnp.asarray(np.frombuffer(policy.parameters, np.float32).copy())
        state = jnp.zeros((args.batch_size, policy.policy.state_words), jnp.float32)
        terminals = jnp.zeros((args.batch_size, 1), jnp.float32)
        first = make_batch(data, train_rows[:args.batch_size], args.batch_size, policy.policy.input_size)
        policy.policy._forward_arrays(initial, state, first[0], terminals, args.batch_size, 1, True)
        core = policy.policy.device_compiled[(args.batch_size, 1)]

        def loss_and_metrics(parameters, obs, mask, label, weight):
            logits = core(parameters, state, obs, terminals)[3][:, 0, :-1]
            log_probs = jax.nn.log_softmax(jnp.where(mask, logits, -1e9), axis=-1)
            chosen = jnp.take_along_axis(log_probs, label[:, None], axis=1)[:, 0]
            nll = -(chosen * weight).sum() / jnp.maximum(weight.sum(), 1)
            anchor_loss = args.anchor * jnp.mean(jnp.square(parameters - initial))
            picked = jnp.argmax(jnp.where(mask, logits, -1e9), axis=-1)
            accuracy = ((picked == label) * weight).sum() / jnp.maximum(weight.sum(), 1)
            return nll + anchor_loss, (nll, accuracy,
                                        ((picked == 3528) * weight).sum() / jnp.maximum(weight.sum(), 1),
                                        (((picked >= 1764) & (picked < 3528)) * weight).sum()
                                        / jnp.maximum(weight.sum(), 1))

        step_fn = jax.jit(jax.value_and_grad(loss_and_metrics, has_aux=True))
        eval_fn = jax.jit(loss_and_metrics)
        params = initial
        moment = jnp.zeros_like(params)
        variance = jnp.zeros_like(params)
        count = 0
        rng = np.random.default_rng(args.seed)
        args.output.mkdir(parents=True, exist_ok=True)

        def evaluate(rows):
            weighted = np.zeros(4, np.float64)
            for start in range(0, len(rows), args.batch_size):
                chunk = rows[start:start+args.batch_size]
                batch = make_batch(data, chunk, args.batch_size, policy.policy.input_size)
                _, metrics = eval_fn(params, *batch)
                weighted += len(chunk) * np.asarray([float(x) for x in metrics])
            return (weighted / len(rows)).tolist()

        best = float("inf")
        history = []
        for epoch in range(args.epochs + 1):
            training = evaluate(train_rows)
            validation = evaluate(validation_rows)
            record = {"epoch": epoch, "train_nll": training[0], "train_top1": training[1],
                      "validation_nll": validation[0], "validation_top1": validation[1],
                      "validation_pass_rate": validation[2], "validation_half_rate": validation[3]}
            history.append(record)
            if validation[0] < best:
                best = validation[0]
                raw = np.asarray(params, np.float32).tobytes()
                (args.output / "best.bin").write_bytes(raw)
                record["best_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
            print(json.dumps(record), flush=True)
            if epoch == args.epochs:
                break
            shuffled = rng.permutation(train_rows)
            for start in range(0, len(shuffled), args.batch_size):
                batch = make_batch(data, shuffled[start:start+args.batch_size], args.batch_size, policy.policy.input_size)
                (_, _), grad = step_fn(params, *batch)
                norm = jnp.linalg.norm(grad)
                grad = grad * jnp.minimum(1, 1 / (norm + 1e-6))
                count += 1
                moment = 0.9 * moment + 0.1 * grad
                variance = 0.999 * variance + 0.001 * jnp.square(grad)
                corrected = moment / (1 - 0.9**count)
                corrected_variance = variance / (1 - 0.999**count)
                params = params - args.learning_rate * corrected / (jnp.sqrt(corrected_variance) + 1e-8)
        (args.output / "training.json").write_text(json.dumps({
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "source_checkpoint_sha256": args.sha256,
            "expert_policy_version_id": metadata["expert_policy_version_id"],
            "train_games": len(train_games), "validation_games": len(validation_games),
            "train_samples": len(train_rows), "validation_samples": len(validation_rows),
            "batch_size": args.batch_size, "learning_rate": args.learning_rate, "anchor": args.anchor,
            "seed": args.seed, "best_validation_nll": best, "history": history,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()

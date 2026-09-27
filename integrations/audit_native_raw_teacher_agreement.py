"""Compare raw native PPO checkpoints on the same legal teacher trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--turns", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1386)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    assert args.games > 0 and args.turns > 0
    manifest = json.loads(args.build.read_text())
    options = manifest["config"]["python_environment"]["options"].copy()
    assert options["coworld_classic"] and not options["hint_features"]
    options.update(
        parallel_games=args.games,
        opponent="random",
        teacher="expander_harvester",
        sparse_teacher=True,
        imitation_weight=0.0,
        supervise_teacher=False,
        teacher_rollouts=False,
    )
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=args.seed, index=0, mode="train", output=Path("/tmp")),
        **options,
    )
    checkpoints = []
    for path in args.checkpoint:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        policy = NativePufferPolicy(args.build, args.training, path, digest)
        assert policy.layers == 0
        checkpoints.append((path.name, digest, policy))
    observations = env.reset(f"teacher-audit-{args.seed}")
    totals = {
        name: {"source_top1": 0, "joint_top1": 0, "source_prob": 0.0, "joint_prob": 0.0}
        for name, _, _ in checkpoints
    }
    count = 0
    try:
        for _ in range(args.turns):
            labels = np.asarray(observations.replay_metadata, dtype=np.int32)
            active = (~env.finished) & (labels[:, 0] >= 0)
            masks = np.asarray(observations.action_masks, dtype=bool)
            assert masks[np.arange(args.games)[active], labels[active, 0]].all()
            values = jnp.asarray(observations.values, dtype=jnp.float32)
            masked = jnp.asarray(masks)
            row = np.arange(args.games)[active]
            source = labels[active, 0]
            split = np.where(labels[active, 1] >= 0, labels[active, 1], 0)
            for name, _, policy in checkpoints:
                logits, _ = policy.forward(values, policy.initial_state(args.games))
                logits = jnp.where(masked, logits[:, :1767], -jnp.inf)
                source_probs = np.asarray(jax.nn.softmax(logits[:, :1765], axis=-1))
                split_probs = np.asarray(jax.nn.softmax(logits[:, 1765:], axis=-1))
                source_choice = np.asarray(jnp.argmax(logits[:, :1765], axis=-1))
                split_choice = np.asarray(jnp.argmax(logits[:, 1765:], axis=-1))
                selected_source_prob = source_probs[row, source]
                selected_split_prob = split_probs[row, split]
                total = totals[name]
                total["source_top1"] += int((source_choice[active] == source).sum())
                total["joint_top1"] += int(((source_choice[active] == source) & (split_choice[active] == split)).sum())
                total["source_prob"] += float(selected_source_prob.sum())
                total["joint_prob"] += float((selected_source_prob * selected_split_prob).sum())
            count += len(row)
            chosen = np.where(labels[:, 0] >= 0, labels[:, 0], 1764)
            split = np.where(labels[:, 1] >= 0, labels[:, 1], 0)
            observations = env.step(np.stack((chosen, split), axis=1)).observation
    finally:
        env.close()
    assert count > 0
    print(json.dumps({
        "scope": "Same teacher-driven Classic states; raw native actor; diagnostic only",
        "games": args.games,
        "turns": args.turns,
        "labeled_decisions": count,
        "policies": {
            name: {"sha256": digest, **{key: value / count for key, value in totals[name].items()}}
            for name, digest, _ in checkpoints
        },
    }), flush=True)


if __name__ == "__main__":
    main()

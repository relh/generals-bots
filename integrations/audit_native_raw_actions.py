"""Audit native greedy actions on each raw policy's own Classic trajectories."""

import argparse
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from metta_training.environment import EnvironmentContext

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment
from integrations.native_puffer_policy import NativePufferPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    options = json.loads(Path(config["environment_build"]).read_text())["config"]["python_environment"]["options"]
    assert options["coworld_classic"] and not options["hint_features"]
    options.update(
        parallel_games=config["games"], opponent="strong_mixed", supervise_teacher=False, teacher_rollouts=False
    )
    result = dict(
        scope="Own greedy trajectories on raw public Classic observations; diagnostic, not held-out wins",
        config=config,
        policies={},
    )
    for name, reference in config["policies"].items():
        policy = NativePufferPolicy(
            **{key: value if key == "sha256" else Path(value) for key, value in reference.items()}
        )
        output = args.output / name
        output.mkdir()
        env = BatchedGeneralsPufferEnvironment(
            context=EnvironmentContext(seed=config["seed"], index=0, mode="evaluate", output=output), **options
        )
        state = policy.initial_state(config["games"])
        observation = env.reset(str(config["seed"]))
        windows = []
        try:
            for start in range(0, config["turns"], 50):
                counts = Counter()
                move_counts = Counter()
                for turn in range(start, min(start + 50, config["turns"])):
                    active = ~env.finished.copy()
                    if not active.any():
                        break
                    values = np.asarray(observation.values, dtype=np.float32)
                    masks = np.asarray(observation.action_masks, dtype=bool)
                    decoded, state = policy.forward(jnp.asarray(values), state)
                    logits = jnp.where(jnp.asarray(masks), decoded[:, :1767], -jnp.inf)
                    moves = jax.nn.softmax(logits[:, :1765], axis=-1)
                    choices = np.asarray(
                        jnp.stack(
                            (jnp.argmax(logits[:, :1765], axis=-1), jnp.argmax(logits[:, 1765:], axis=-1)), axis=-1
                        ),
                        dtype=np.int32,
                    )
                    probabilities = np.asarray(moves)
                    assert masks[np.arange(config["games"]), choices[:, 0]].all()
                    assert masks[np.arange(config["games"]), 1765 + choices[:, 1]].all()
                    legal_moves = masks[:, :1764].sum(axis=1)
                    move_entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(axis=1)
                    counts["decisions"] += int(active.sum())
                    counts["passes"] += int((active & (choices[:, 0] == 1764)).sum())
                    counts["passes_with_moves"] += int((active & (legal_moves > 0) & (choices[:, 0] == 1764)).sum())
                    counts["movable"] += int((active & (legal_moves > 0)).sum())
                    counts["split_one"] += int((active & (choices[:, 1] == 1)).sum())
                    counts["legal_moves"] += int(legal_moves[active].sum())
                    counts["entropy_milli"] += int(np.rint(move_entropy[active].sum() * 1000))
                    counts["max_probability_milli"] += int(np.rint(probabilities[active].max(axis=1).sum() * 1000))
                    move_counts.update(map(int, choices[active, 0]))
                    observation = env.step(choices).observation
                    if policy.layers:
                        done = np.asarray(env.finished)
                        state = jnp.where(jnp.asarray(done)[None, :, None], 0, state)
                if not counts["decisions"]:
                    break
                total = counts["decisions"]
                windows.append(
                    dict(
                        first_turn=start,
                        last_turn=min(start + 49, config["turns"] - 1),
                        decisions=total,
                        pass_fraction=counts["passes"] / total,
                        pass_with_moves_fraction=(
                            counts["passes_with_moves"] / counts["movable"] if counts["movable"] else 0
                        ),
                        split_one_fraction=counts["split_one"] / total,
                        legal_moves_mean=counts["legal_moves"] / total,
                        move_entropy=counts["entropy_milli"] / (1000 * total),
                        move_max_probability=counts["max_probability_milli"] / (1000 * total),
                        top_moves=move_counts.most_common(5),
                    )
                )
        finally:
            env.close()
        result["policies"][name] = dict(checkpoint_sha256=reference["sha256"], windows=windows)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["policies"]), flush=True)


if __name__ == "__main__":
    main()

"""Compare verified checkpoints on identical, player-visible Classic trajectories."""

import argparse
import json
from pathlib import Path

import jax
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
    policies = {
        name: NativePufferPolicy(**{key: Path(value) if key != "sha256" else value for key, value in reference.items()})
        for name, reference in config["policies"].items()
    }
    states = {name: policy.initial_state(config["games"]) for name, policy in policies.items()}
    options = json.loads(Path(config["environment_build"]).read_text())["config"]["python_environment"]["options"]
    options.update(parallel_games=config["games"], supervise_teacher=False, teacher_rollouts=False)
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=config["seed"], index=0, mode="train", output=args.output), **options
    )
    observation = env.reset(str(config["seed"]))
    seats = list(range(config["games"]))
    samples = {name: [] for name in policies}
    parameter_reports = {}
    baseline = policies[config["parent"]]
    for name, policy in policies.items():
        report = {}
        for key in ("encoder", "decoder", "recurrent"):
            current = np.asarray(getattr(policy, key))
            original = np.asarray(getattr(baseline, key))
            delta = current - original
            report[key] = dict(
                parameters=int(delta.size),
                rms_change=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
                max_change=float(np.max(np.abs(delta))) if delta.size else 0.0,
            )
        parameter_reports[name] = report
    try:
        for turn in range(config["turns"]):
            planes = np.asarray(observation.values).reshape(len(seats), 14, 441)
            source = planes[:, 4:8].reshape(len(seats), 1764).argmax(axis=1)
            passing = planes[:, 3, 0] > 0
            hint = np.stack((np.where(passing, 1764, source), planes[:, 2, 0] > 0), axis=1)
            legal = np.asarray(observation.action_masks, bool)
            flexible = legal[:, :1765].sum(axis=1) > 1
            greedy = {}
            for name, policy in policies.items():
                decoded, states[name] = policy.forward(
                    jax.numpy.asarray(observation.values, dtype=jax.numpy.float32), states[name],
                )
                assert np.isfinite(np.asarray(decoded)).all()
                logits = jax.numpy.where(jax.numpy.asarray(legal), decoded[:, :1767], -jax.numpy.inf)
                probabilities = np.asarray(jax.numpy.concatenate((
                    jax.nn.softmax(logits[:, :1765], axis=-1),
                    jax.nn.softmax(logits[:, 1765:], axis=-1),
                ), axis=-1), dtype=np.float64)
                moves, splits = probabilities[:, :1765], probabilities[:, 1765:]
                greedy[name] = np.stack((moves.argmax(axis=1), splits.argmax(axis=1)), axis=1)
                assert legal[np.arange(len(seats)), greedy[name][:, 0]].all()
                teacher_legal = legal[np.arange(len(seats)), hint[:, 0]]
                agreement = (greedy[name][:, 0] == hint[:, 0]) & (
                    (hint[:, 0] == 1764) | (greedy[name][:, 1] == hint[:, 1])
                )
                samples[name].append(
                    dict(
                        turn=turn,
                        flexible=flexible,
                        teacher_legal=teacher_legal,
                        agreement=agreement,
                        move_changed_from_hint=greedy[name][:, 0] != hint[:, 0],
                        split_changed_from_hint=(greedy[name][:, 0] == hint[:, 0]) & (
                            hint[:, 0] != 1764) & (greedy[name][:, 1] != hint[:, 1]
                        ),
                        move_entropy=-(moves * np.log(np.maximum(moves, 1e-300))).sum(axis=1),
                        split_entropy=-(splits * np.log(np.maximum(splits, 1e-300))).sum(axis=1),
                        move_max=moves.max(axis=1),
                        pass_prediction=greedy[name][:, 0] == 1764,
                        teacher_probability=moves[np.arange(len(seats)), hint[:, 0]],
                    )
                )
            parent = greedy[config["parent"]]
            for name in policies:
                changed = (greedy[name][:, 0] != parent[:, 0]) | (
                    (greedy[name][:, 0] != 1764) & (greedy[name][:, 1] != parent[:, 1])
                )
                samples[name][-1]["changed_from_parent"] = changed
            # The public Expander hint drives every policy's identical trajectory.
            assert legal[np.arange(len(seats)), hint[:, 0]].all()
            transition = env.step(hint)
            observation = transition.observation
            done = np.asarray(transition.terminated, dtype=bool)
            for name in policies:
                states[name] = jax.numpy.where(jax.numpy.asarray(done)[None, :, None], 0, states[name])
    finally:
        env.close()
    result = dict(
        scope="Identical real training trajectories; diagnostic, not held-out match performance", config=config
    )
    result["parameter_changes"] = parameter_reports
    result["policies"] = {}
    for name, rows in samples.items():
        arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0] if key != "turn"}
        turns = np.repeat(np.arange(config["turns"]), config["games"])
        flexible = arrays["flexible"]
        assert flexible.any()
        result["policies"][name] = dict(
            decisions=int(len(flexible)),
            flexible_decisions=int(flexible.sum()),
            teacher_legal_fraction=float(arrays["teacher_legal"].mean()),
            teacher_agreement=float(arrays["agreement"][flexible].mean()),
            move_changed_from_hint=float(arrays["move_changed_from_hint"][flexible].mean()),
            same_move_wrong_split=float(arrays["split_changed_from_hint"][flexible].mean()),
            hint_agreement_by_turn_window=[
                dict(first_turn=start, last_turn=min(start + 63, config["turns"] - 1),
                     decisions=int((flexible & (turns >= start) & (turns < start + 64)).sum()),
                     agreement=float(arrays["agreement"][flexible & (turns >= start) & (turns < start + 64)].mean()))
                for start in range(0, config["turns"], 64)
            ],
            changed_from_parent=float(arrays["changed_from_parent"][flexible].mean()),
            predicted_pass_fraction=float(arrays["pass_prediction"][flexible].mean()),
            move_entropy=float(arrays["move_entropy"][flexible].mean()),
            split_entropy=float(arrays["split_entropy"][flexible].mean()),
            teacher_move_probability=float(arrays["teacher_probability"][flexible].mean()),
            move_max_quantiles=np.quantile(arrays["move_max"][flexible], [0.1, 0.5, 0.9]).tolist(),
        )
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["policies"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Compare verified checkpoints on identical, player-visible Classic trajectories."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    policies = {
        name: FrozenPolicy(FrozenPolicyConfig(**reference, device="cuda:0"))
        for name, reference in config["policies"].items()
    }
    for policy in policies.values():
        policy.reset(str(config["seed"]))
    options = json.loads(Path(config["environment_build"]).read_text())["config"]["python_environment"]["options"]
    options.update(parallel_games=config["games"], supervise_teacher=False, teacher_rollouts=False)
    env = BatchedGeneralsPufferEnvironment(
        context=EnvironmentContext(seed=config["seed"], index=0, mode="train", output=args.output), **options
    )
    observation = env.reset(str(config["seed"]))
    seats = list(range(config["games"]))
    samples = {name: [] for name in policies}
    parameter_reports = {}
    for name, policy in policies.items():
        buffers = policy.policy.buffers
        named = jax.tree.leaves_with_path(buffers.fn.params(buffers.template))
        weights = np.frombuffer(policy.parameters, np.float32)
        initial = np.frombuffer(policy.policy.initial_parameters, np.float32)
        reports = []
        for (path, _), spec in zip(named, buffers.parameters, strict=True):
            span = slice(spec.offset, spec.offset + spec.size)
            prior = np.isclose(initial[span], 8.0, rtol=0, atol=1e-7)
            if prior.any():
                selected = weights[span][prior]
                reports.append(
                    dict(
                        path=str(path),
                        count=int(prior.sum()),
                        mean=float(selected.mean()),
                        minimum=float(selected.min()),
                        maximum=float(selected.max()),
                    )
                )
        parameter_reports[name] = reports
    try:
        for turn in range(config["turns"]):
            planes = np.asarray(observation.values).reshape(len(seats), 14, 441)
            source = planes[:, 4:8].reshape(len(seats), 1764).argmax(axis=1)
            passing = planes[:, 3, 0] > 0
            hint = np.stack((np.where(passing, 1764, source), planes[:, 2, 0] > 0), axis=1)
            legal = np.asarray(observation.action_masks, bool)
            flexible = legal[:, :1765].sum(axis=1) > 1
            greedy = {}
            predictions = {}
            for name, policy in policies.items():
                scale = config["hint_scales"][name]
                view = observation
                if scale != 1:
                    values = np.asarray(observation.values).copy().reshape(len(seats), 14, 441)
                    values[:, 2:8] *= scale
                    view = observation.model_copy(update={"values": values.reshape(len(seats), 6174)})
                predictions[name] = policy.predict_many(seats, view)
                probabilities = np.asarray([target.probabilities for target in predictions[name]])
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
                        move_entropy=-(moves * np.log(np.maximum(moves, 1e-300))).sum(axis=1),
                        split_entropy=-(splits * np.log(np.maximum(splits, 1e-300))).sum(axis=1),
                        move_max=moves.max(axis=1),
                        teacher_probability=moves[np.arange(len(seats)), hint[:, 0]],
                    )
                )
            parent = greedy[config["parent"]]
            for name in policies:
                changed = (greedy[name][:, 0] != parent[:, 0]) | (
                    (greedy[name][:, 0] != 1764) & (greedy[name][:, 1] != parent[:, 1])
                )
                samples[name][-1]["changed_from_parent"] = changed
            actions = [policies[config["driver"]].sample(target) for target in predictions[config["driver"]]]
            transition = env.step(actions)
            observation = transition.observation
            for seat, done in zip(seats, transition.terminated, strict=True):
                if done:
                    for policy in policies.values():
                        policy.states.pop(seat, None)
    finally:
        env.close()
    result = dict(
        scope="Identical real training trajectories; diagnostic, not held-out match performance", config=config
    )
    result["hint_parameters"] = parameter_reports
    result["policies"] = {}
    for name, rows in samples.items():
        arrays = {key: np.concatenate([row[key] for row in rows]) for key in rows[0] if key != "turn"}
        flexible = arrays["flexible"]
        assert flexible.any()
        result["policies"][name] = dict(
            decisions=int(len(flexible)),
            flexible_decisions=int(flexible.sum()),
            teacher_legal_fraction=float(arrays["teacher_legal"].mean()),
            teacher_agreement=float(arrays["agreement"][flexible].mean()),
            changed_from_parent=float(arrays["changed_from_parent"][flexible].mean()),
            move_entropy=float(arrays["move_entropy"][flexible].mean()),
            split_entropy=float(arrays["split_entropy"][flexible].mean()),
            teacher_move_probability=float(arrays["teacher_probability"][flexible].mean()),
            move_max_quantiles=np.quantile(arrays["move_max"][flexible], [0.1, 0.5, 0.9]).tolist(),
        )
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["policies"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Score a saved flat Fabric actor on hosted expert public observations."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != args.sha256:
        raise ValueError("Checkpoint SHA256 mismatch")
    dataset = np.load(args.dataset)
    metadata = json.loads(str(dataset["metadata"]))
    policy = FrozenPolicy(FrozenPolicyConfig(
        build=args.build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0",
    ))
    actions = dataset["actions"]
    observations = dataset["observations"]
    masks = np.unpackbits(dataset["action_masks"], axis=1, count=3529).astype(bool)
    if observations.shape[1] != policy.policy.observation_size or policy.policy.action_sizes != [3529]:
        raise ValueError("Dataset and policy codec differ")
    predictions = []
    expert_probabilities = []
    expert_ranks = []
    for start in range(0, len(actions), args.batch_size):
        end = min(start + args.batch_size, len(actions))
        values = observations[start:end].astype(np.float32)
        transported = np.zeros((end - start, policy.policy.input_size), np.float32)
        transported[:, :values.shape[1]] = values
        state = bytes((end - start) * policy.policy.state_words * 4)
        with jax.default_device(policy.device):
            output, _, _ = policy.policy.forward(
                policy.parameters, state, transported.tobytes(), bytes(4 * (end - start)),
                end - start, 1, True,
            )
        logits = np.frombuffer(output, np.float32).reshape(end - start, policy.policy.output_size)[:, :-1]
        legal_logits = np.where(masks[start:end], logits, -np.inf)
        labels = actions[start:end]
        selected = legal_logits[np.arange(end-start), labels]
        if not np.isfinite(selected).all():
            raise ValueError("An expert label is illegal")
        shifted = legal_logits - legal_logits.max(axis=1, keepdims=True)
        expert_probabilities.extend(np.exp(shifted[np.arange(end-start), labels]) /
                                    np.exp(shifted).sum(axis=1))
        expert_ranks.extend(1 + (legal_logits > selected[:, None]).sum(axis=1))
        predictions.extend(legal_logits.argmax(axis=1))
    predictions = np.asarray(predictions)
    probabilities = np.asarray(expert_probabilities)
    ranks = np.asarray(expert_ranks)
    turns = dataset["turn"]
    report = {"checkpoint_sha256": args.sha256, "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "expert_policy_version_id": metadata["expert_policy_version_id"], "sample_count": len(actions),
              "phases": {}}
    for name, lo, hi in (("early_0_99", 0, 100), ("middle_100_199", 100, 200),
                         ("late_200_plus", 200, 1201)):
        for split in ("train", "holdout"):
            game_ids = [i for i, episode in enumerate(metadata["episodes"]) if episode["split"] == split]
            rows = np.flatnonzero((turns >= lo) & (turns < hi) & np.isin(dataset["episode"], game_ids))
            if len(rows) == 0:
                continue
            predicted = predictions[rows]
            expert = actions[rows]
            report["phases"][f"{split}_{name}"] = {
                "samples": len(rows), "top1_agreement": float(np.mean(predicted == expert)),
                "top10_agreement": float(np.mean(ranks[rows] <= 10)),
                "mean_expert_rank": float(np.mean(ranks[rows])),
                "mean_expert_probability": float(np.mean(probabilities[rows])),
                "predicted_pass_rate": float(np.mean(predicted == 3528)),
                "expert_pass_rate": float(np.mean(expert == 3528)),
                "predicted_half_rate": float(np.mean((predicted >= 1764) & (predicted < 3528))),
                "expert_half_rate": float(np.mean((expert >= 1764) & (expert < 3528))),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

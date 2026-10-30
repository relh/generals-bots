"""Check that a one-seat checkpoint has identical inference in the self-play build."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from metta_training.environment import EnvironmentContext
from metta_training.inference import FrozenPolicy
from metta_training.model_config import FrozenPolicyConfig

from integrations.metta_puffer import BatchedGeneralsPufferEnvironment


def logits_many(policy: FrozenPolicy, observation) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observation.values, dtype=np.float32)
    masks = np.asarray(observation.action_masks, dtype=bool)
    assert values.shape == (16, policy.policy.observation_size)
    assert masks.shape == (16, sum(policy.policy.action_sizes))
    transported = np.zeros((16, policy.policy.input_size), dtype=np.float32)
    transported[:, :values.shape[1]] = values
    with jax.default_device(policy.device):
        outputs, _, _ = policy.policy.forward(
            policy.parameters, bytes(16 * policy.policy.state_words * 4),
            transported.tobytes(), bytes(16 * 4), 16, 1, True,
        )
    logits = np.frombuffer(outputs, np.float32).reshape(16, policy.policy.output_size)[:, :-1]
    actions = []
    offset = 0
    for size in policy.policy.action_sizes:
        actions.append(np.argmax(np.where(masks[:, offset:offset + size],
                                          logits[:, offset:offset + size], -np.inf), axis=1))
        offset += size
    return np.stack(actions, axis=1), logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--target-build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.sha256
    source = json.loads(args.source_build.read_text())
    target = json.loads(args.target_build.read_text())
    assert source["revision"] == target["revision"]
    assert source["model_state_words"] == target["model_state_words"] == 22956
    for key in ("factory", "observation_size", "action_sizes", "options"):
        assert source["config"]["fabric"][key] == target["config"]["fabric"][key]
    args.output.mkdir(parents=True, exist_ok=False)
    options = dict(source["config"]["python_environment"]["options"])
    options.update(parallel_games=16, coworld_pool_size=16,
                   supervise_teacher=False, deduplicate_opponent_branches=False)
    context = EnvironmentContext(seed=1389, index=0, mode="evaluate", output=args.output)
    environment = BatchedGeneralsPufferEnvironment(context=context, **options)
    try:
        observation = environment.reset("1389:0:0")
        policies = [FrozenPolicy(FrozenPolicyConfig(
            build=build, checkpoint=args.checkpoint, sha256=args.sha256, device="cuda:0"
        )) for build in (args.source_build, args.target_build)]
        for policy in policies:
            policy.reset("1389:0:0")
        actions, logits = zip(*(logits_many(policy, observation) for policy in policies))
        assert np.array_equal(actions[0], actions[1])
        assert np.array_equal(logits[0], logits[1])
        result = dict(source_model_sha256=source["model_sha256"],
                      target_model_sha256=target["model_sha256"],
                      checkpoint_sha256=args.sha256, games=16,
                      logits_shape=list(logits[0].shape),
                      logits_equal=True, actions_equal=True)
        (args.output / "parity.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()

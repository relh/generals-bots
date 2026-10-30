"""Initialize the pinned Classic actor from one verified corridor checkpoint.

Adds a single explicit environment-transfer equivalence. The ordinary model,
checkpoint digest, float finiteness and learner-state guards still apply.
"""

import hashlib
import importlib.util
import runpy
import sys
from pathlib import Path

import jax


def verified_sequence_curriculum_transfer(source, target, checkpoint_sha256):
    source_env, target_env = source.config.python_environment, target.config.python_environment
    if source_env is None or target_env is None:
        return False
    return (
        checkpoint_sha256 == "76762ef01e9b115e9ae6b5d97daba5281ff4fe7e4c8d5646f5929fd95e21dc3d"
        and source.environment_sha256 == "eca17b68999d49d046ab21273c17f1623faa4d4a6133714432b3c7578ce19b6b"
        and target.environment_sha256 == "78d56222050c4ac980ed33e6f2910f2a170157198cc502cc2a13e75f765d76f8"
        and source.model_sha256 == target.model_sha256
        == "c0046141f74f771e8eba6b5296f04913f8736eae6803dab717b90a49fe8b161d"
        and source.model_state_words == target.model_state_words == 23834
        and source.revision == target.revision == "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
        and source_env.factory == "audit_puffer_sequence_credit:SequenceCreditEnvironment"
        and target_env.factory == "integrations.metta_puffer:BatchedGeneralsPufferEnvironment"
        and source_env.spec == target_env.spec
        and source_env.spec.observation_size == 4851
        and source_env.spec.action_sizes == [3529]
        and source_env.spec.agents == 4096
        and not source_env.spec.teacher
        and source.config.model_dump(exclude={"python_environment"})
        == target.config.model_dump(exclude={"python_environment"})
    )


def main():
    source = Path(__file__).with_name("puffer_coworld_frozen_transfer.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "4d18c06c59b4dad321bf61ba4d4aed552dc406159c96072e880cb518162ecea6"
    )
    if jax.devices()[0] not in jax.devices("cuda") or not jax.devices("cpu"):
        raise RuntimeError("Puffer training requires CUDA and CPU JAX backends")
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.verified_classic_flat_scripted_transfer
    module.verified_classic_flat_scripted_transfer = lambda source, target, digest: (
        original(source, target, digest) or verified_sequence_curriculum_transfer(source, target, digest)
    )
    runpy.run_module("metta_training.cli", run_name="__main__")


if __name__ == "__main__":
    main()

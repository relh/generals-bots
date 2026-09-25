"""Stage an isolated Puffer runtime for a verified supervised-to-PPO pilot.

The upstream runtime rejects policy-only initialization when the loss changes.
This experiment permits only the known 31.46M sparse-teacher source model and
the matching 8,004-parameter PPO target, while retaining every other check.
It never edits the shared runtime tree.
"""

import hashlib
import shutil
import sys
from pathlib import Path


EXPECTED_SOURCE = "25ab571e3083a0a89c857970b5ae37d1c2e43e980e9eda649286a604cad4206e"
OLD = '''        if reference.allow_environment_transfer:
            source_env = source.build.config.python_environment
            target_env = manifest.config.python_environment
            compatible_environment = (
                source_env is not None
                and target_env is not None
                and source_env.factory == target_env.factory
                and source_env.spec == target_env.spec
                and source.build.config.model_dump(exclude={"python_environment"})
                == manifest.config.model_dump(exclude={"python_environment"})
                and source.build.revision == manifest.revision
            )
        else:
            compatible_environment = (
                source.build.config == manifest.config
                and source.build.environment_sha256 == manifest.environment_sha256
            )
        if (
            not compatible_environment
            or source.build.model_sha256 != manifest.model_sha256
            or source.build.model_state_words != manifest.model_state_words
        ):
            raise ValueError("Initialization requires matching model and compatible environment configuration")
'''
NEW = '''        loss_transfer = False
        if reference.allow_environment_transfer and not reference.restore_learner:
            source_build = source.build.config.model_dump(exclude={"python_environment"})
            target_build = manifest.config.model_dump(exclude={"python_environment"})
            source_losses = source_build["fabric"].pop("losses")
            target_losses = target_build["fabric"].pop("losses")
            loss_transfer = (
                source.build.model_sha256 == "c199a856c706ad2d859ff0ea1091d191ff7922e3ffc08af2097383b1afbf55c9"
                and manifest.model_sha256 == "d8c86c6207159518e4465879f1e19de55857260e00744c7d44f35fa550140125"
                and reference.sha256 == "939e71adc493980122badd44939944d4af470b15dd52c696ee3b55425811a499"
                and source_build == target_build
                and len(source_losses) == len(target_losses) == 1
                and source_losses[0]["name"] == target_losses[0]["name"] == "sparse_teacher"
                and source_losses[0]["factory"] == target_losses[0]["factory"]
                and source_losses[0]["outputs"] == target_losses[0]["outputs"] == 0
                and source_losses[0]["assets"] == target_losses[0]["assets"] == []
                and source_losses[0]["options"]["action_sizes"] == target_losses[0]["options"]["action_sizes"]
                and source_losses[0]["replace_ppo"] is True
                and target_losses[0]["replace_ppo"] is False
                and source_losses[0]["options"]["coefficient"] == 1.0
                and target_losses[0]["options"]["coefficient"] == 0.25
            )
        if reference.allow_environment_transfer:
            source_env = source.build.config.python_environment
            target_env = manifest.config.python_environment
            compatible_environment = (
                source_env is not None
                and target_env is not None
                and source_env.factory == target_env.factory
                and source_env.spec == target_env.spec
                and (source.build.config.model_dump(exclude={"python_environment"})
                     == manifest.config.model_dump(exclude={"python_environment"}) or loss_transfer)
                and source.build.revision == manifest.revision
            )
        else:
            compatible_environment = (
                source.build.config == manifest.config
                and source.build.environment_sha256 == manifest.environment_sha256
            )
        if (
            not compatible_environment
            or (source.build.model_sha256 != manifest.model_sha256 and not loss_transfer)
            or source.build.model_state_words != manifest.model_state_words
        ):
            raise ValueError("Initialization requires matching model and compatible environment configuration")
'''


def main() -> None:
    source, destination = map(Path, sys.argv[1:])
    original = source / "metta_training" / "puffer.py"
    data = original.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_SOURCE:
        raise ValueError("Unexpected upstream Puffer runtime source")
    text = data.decode()
    if text.count(OLD) != 1:
        raise ValueError("Expected one exact initialization guard")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "metta_training", destination / "metta_training")
    patched = destination / "metta_training" / "puffer.py"
    patched.write_text(text.replace(OLD, NEW))
    print("source_sha256=" + EXPECTED_SOURCE)
    print("patched_sha256=" + hashlib.sha256(patched.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()

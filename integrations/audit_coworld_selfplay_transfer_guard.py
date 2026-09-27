"""Verify the pinned Puffer policy-transfer exception accepts one exact lineage."""

import importlib.util
import json
import sys
from pathlib import Path


def main():
    root = Path("/recovery")
    source = root / "puffer_selfplay_policy_transfer-v1-20260927.py"
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source_build = module.BuildManifest.model_validate_json((
        root / "classic-spatial4-hintcurriculum-prior05-32m-build-27745/build.json"
    ).read_text())
    target_build = module.BuildManifest.model_validate_json((
        root / "classic-selfplay-teacher-h128-build-27857/build.json"
    ).read_text())
    checkpoint_sha256 = "c6cc6db63b3b5855d1ab7f38b99baf8ed25cfcb51f2c34819a43a1d58b51fe3a"
    verifier = module.verified_classic_selfplay_policy_transfer
    assert verifier(source_build, target_build, checkpoint_sha256)
    assert not verifier(source_build, target_build, "0" * 64)
    wrong_factory = target_build.model_copy(update={"config": target_build.config.model_copy(update={
        "python_environment": target_build.config.python_environment.model_copy(update={
            "factory": "integrations.metta_puffer:BatchedGeneralsPufferEnvironment"
        })
    })})
    assert not verifier(source_build, wrong_factory, checkpoint_sha256)
    wrong_model = target_build.model_copy(update={"model_sha256": "0" * 64})
    assert not verifier(source_build, wrong_model, checkpoint_sha256)
    print(json.dumps({"accepted": checkpoint_sha256, "rejected_bad_digest": True,
                      "rejected_bad_factory": True, "rejected_bad_model": True}), flush=True)


if __name__ == "__main__":
    main()

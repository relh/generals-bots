"""Verify the pinned Puffer policy-transfer exception accepts one exact lineage."""

import importlib.util
import json
import sys
from pathlib import Path


def main():
    root = Path("/recovery")
    source = root / "puffer_selfplay_iter1_transfer-v1-20260927.py"
    spec = importlib.util.spec_from_file_location("metta_training.puffer", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    source_build = module.BuildManifest.model_validate_json((
        root / "classic-selfplay-teacher-h128-build-27857/build.json"
    ).read_text())
    target_build = module.BuildManifest.model_validate_json((
        root / "classic-selfplay-iter1-build-28048/build/build.json"
    ).read_text())
    checkpoint_sha256 = "e9c909e4f8143a66192686db2f8891dcab2d9144af38f0c0fde4211b770817cf"
    verifier = module.verified_classic_iter1_calibrated_transfer
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
    wrong_scale = target_build.model_copy(update={"config": target_build.config.model_copy(update={
        "python_environment": target_build.config.python_environment.model_copy(update={
            "options": target_build.config.python_environment.options | {"move_hint_scale": 0.5}
        })
    })})
    assert not verifier(source_build, wrong_scale, checkpoint_sha256)
    print(json.dumps({"accepted": checkpoint_sha256, "rejected_bad_digest": True,
                      "rejected_bad_factory": True, "rejected_bad_model": True,
                      "rejected_bad_scale": True}), flush=True)


if __name__ == "__main__":
    main()

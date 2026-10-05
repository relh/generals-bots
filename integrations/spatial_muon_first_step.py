"""Capture two startup callbacks to verify the native optimizer's first step.

Enable only with METTA_SPATIAL_MUON_STEP_AUDIT_DIR. These synchronous host
copies are confined to startup and must be excluded from throughput warmup.
"""

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def install(native_module):
    from integrations.direct_spatial_optimization import DirectTape

    cls = native_module.NativeFabricPolicy
    if not getattr(cls, "_generals_direct_spatial", False):
        raise RuntimeError("First-step audit requires the installed direct spatial adapter")
    if getattr(cls, "_generals_muon_step_audit", False):
        raise RuntimeError("First-step audit is already installed")
    original = cls.backward_device_arrays
    directory = Path(os.environ["METTA_SPATIAL_MUON_STEP_AUDIT_DIR"])
    directory.mkdir(parents=True, exist_ok=False)
    calls = 0

    def save(name, array):
        data = np.asarray(array, dtype=np.float32).tobytes()
        with (directory / name).open("xb") as stream:
            stream.write(data)
        return hashlib.sha256(data).hexdigest()

    @functools.wraps(original)
    def backward(self, tape, logits, values):
        nonlocal calls
        if not isinstance(tape, DirectTape):
            return original(self, tape, logits, values)
        calls += 1
        if calls > 2:
            return original(self, tape, logits, values)
        digest = save("before.bin" if calls == 1 else "after.bin", tape.parameters)
        if calls == 1:
            inputs = {}
            for label, array in (("observations", tape.observations),
                                 ("logits_cotangent", logits), ("values_cotangent", values)):
                inputs[label] = hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
        gradient = original(self, tape, logits, values)
        if calls == 1:
            receipt = dict(before_sha256=digest, gradient_sha256=save("gradient.bin", gradient),
                           inputs=inputs, blocks=self.spatial_optimizer_layout_report,
                           orientation=os.environ.get("METTA_SPATIAL_MUON_DENSE_ORIENTATION", "storage"))
            receipt["context_matrix"] = os.environ.get("METTA_SPATIAL_MUON_CONTEXT_MATRIX", "0") == "1"
            with (directory / "receipt.json").open("x") as stream:
                json.dump(receipt, stream, indent=2)
        else:
            with (directory / "after-sha256.txt").open("x") as stream:
                stream.write(digest + "\n")
        return gradient

    cls.backward_device_arrays = backward
    cls._generals_muon_step_audit = True

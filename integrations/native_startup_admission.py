"""Admit the actual native actor before loading checkpoint bytes, without rebuilding it."""

import hashlib
import importlib
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bindings(record_path):
    from integrations.native_spatial_asset import load_asset, training_contract
    from metta_training.model_config import FabricConfig
    from metta_training.native_build import fabric_fingerprint

    record_path = Path(record_path)
    record = json.loads(record_path.read_text())
    build, config = record["build"], record["config"]
    fabric = FabricConfig.model_validate(build["config"]["fabric"])
    factory = Path(importlib.import_module(fabric.factory.split(":", 1)[0]).__file__)
    if fabric_fingerprint(fabric) != build["model_sha256"] or build["model_state_words"] != 0:
        raise ValueError("Native admission model or external state differs")
    if build["native_admission_sha256"] != digest(__file__):
        raise ValueError("Build omitted or changed native admission")
    reference = config.get("initialize")
    initial = record_path.parent / "initial-policy.bin"
    learner = initial.with_name(initial.name + ".learner")
    asset = load_asset(Path(reference["asset"]), manifest_sha256=reference["manifest_sha256"]) if reference else None
    if asset:
        if (
            asset.metadata["factory_source_sha256"] != digest(factory)
            or asset.metadata["model_sha256"] != build["model_sha256"]
            or asset.metadata["fabric"] != fabric.model_dump(mode="json")
            or asset.metadata["training_contract"]
            != training_contract(build["config"]["python_environment"]["options"], config["overrides"])
            or initial.read_bytes() != asset.policy
        ):
            raise ValueError("Initializer factory, model, fabric, objective or policy bytes differ")
        if reference["restore_learner"]:
            if learner.read_bytes() != asset.learner or asset.metadata["learner_configuration"] != {
                "seed": config["seed"],
                "overrides": config["overrides"],
            }:
                raise ValueError("Initializer optimizer bytes or configuration differ")
        elif learner.exists():
            raise ValueError("Fresh optimizer unexpectedly has restored state")
    elif initial.exists() or learner.exists():
        raise ValueError("Uninitialized training unexpectedly has checkpoint files")
    return dict(
        schema="generals-native-startup-admission-v1",
        run_sha256=digest(record_path),
        admission_source_sha256=digest(__file__),
        factory_source_sha256=digest(factory),
        model_sha256=build["model_sha256"],
        external_state_words=0,
        initializer_manifest_sha256=reference["manifest_sha256"] if reference else None,
        initial_policy_sha256=digest(initial) if asset else None,
        initial_learner_sha256=digest(learner) if asset and reference["restore_learner"] else None,
        expected_abi_sha256=asset.metadata["abi_sha256"] if asset else None,
        expected_parameter_count=asset.metadata["parameter_count"] if asset else None,
        initialization="restored"
        if asset and reference["restore_learner"]
        else "fresh_optimizer"
        if asset
        else "new_parameters",
    )


def verify_receipt(record_path):
    expected = bindings(record_path)
    receipt = json.loads(Path(record_path).with_name("native-admission.json").read_text())
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Native admission receipt identity differs")
    if (
        type(receipt.get("native_pid")) is not int
        or receipt["native_pid"] <= 0
        or len(receipt.get("actual_abi_sha256", "")) != 64
        or type(receipt.get("actual_parameter_count")) is not int
        or receipt["actual_parameter_count"] <= 0
        or expected["expected_abi_sha256"] is not None
        and receipt["actual_abi_sha256"] != expected["expected_abi_sha256"]
        or expected["expected_parameter_count"] is not None
        and receipt["actual_parameter_count"] != expected["expected_parameter_count"]
    ):
        raise ValueError("Native admission actual layout differs")
    return receipt


def admit(policy):
    from integrations.native_spatial_asset import abi_digest

    record_path = Path(os.environ["METTA_RUN_RECORD"])
    expected = bindings(record_path)
    record = json.loads(record_path.read_text())
    if (
        expected["run_sha256"] != os.environ["METTA_NATIVE_ADMISSION_RUN_SHA256"]
        or expected["admission_source_sha256"] != os.environ["METTA_NATIVE_ADMISSION_SOURCE_SHA256"]
        or json.loads(os.environ["METTA_FABRIC_CONFIG"]) != record["build"]["config"]["fabric"]
        or json.loads(os.environ["METTA_OBJECTIVE_SETTINGS"])
        != {key: value for key, value in record["config"]["overrides"].items() if key.startswith("objective.")}
        or policy.state_words != expected["external_state_words"]
    ):
        raise ValueError("Native actor configuration or sealed request differs")
    initial = record_path.parent / "initial-policy.bin"
    if os.environ.get("METTA_INITIAL_POLICY", "") != (str(initial) if expected["initial_policy_sha256"] else ""):
        raise ValueError("Native checkpoint load path differs")
    if os.environ.get("METTA_INITIAL_LEARNER", "") != (
        str(initial) + ".learner" if expected["initial_learner_sha256"] else ""
    ):
        raise ValueError("Native optimizer load path differs")
    receipt = dict(
        expected,
        native_pid=os.getpid(),
        actual_abi_sha256=abi_digest(policy),
        actual_parameter_count=policy.buffers.parameter_words,
    )
    path = record_path.with_name("native-admission.json")
    with path.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write("\n")
    verify_receipt(record_path)
    print("NATIVE_STARTUP_ADMISSION " + json.dumps(dict(receipt_sha256=digest(path), **receipt)), flush=True)
    return True


def install(source):
    path = Path(source) / "src/pufferl.cu"
    text = path.read_text()
    anchor = (
        '    PuffeRL* pufferl = create_pufferl(ini, ctx);\n    const char* initial = getenv("METTA_INITIAL_POLICY");'
    )
    replacement = """    PuffeRL* pufferl = create_pufferl(ini, ctx);
    PyGILState_STATE admission_gil = PyGILState_Ensure();
    PyObject* admission_module = metta_checked(PyImport_ImportModule("integrations.native_startup_admission"));
    PyObject* admission = metta_checked(PyObject_CallMethod(admission_module, "admit", "O", metta_policy));
    if (admission != Py_True) { fprintf(stderr, "Native startup admission refused\\n"); abort(); }
    Py_DECREF(admission);
    Py_DECREF(admission_module);
    PyGILState_Release(admission_gil);
    const char* initial = getenv("METTA_INITIAL_POLICY");"""
    if text.count(anchor) != 1:
        raise ValueError("Pinned native startup admission anchor changed")
    path.write_text(text.replace(anchor, replacement))

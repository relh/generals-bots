"""Opt in to output/input Muon scaling for two input-major spatial matrices.

The flat checkpoint and Newton–Schulz algebra stay in their existing order.
Transposition commutes with that algebra; only its final aspect-ratio scale
needs the logical output/input dimensions used by Puffer's standard layers.
"""

import functools
import hashlib
import json
from pathlib import Path

INSTALLED_ALGO_SHA256 = "ffce514b8bb9fb7bd36922722ddfd42c90ace6d167004b5b732b8cba710682b0"
CANONICAL_ALGO_SHA256 = "cee5f060a2c381f2a2b245695436c15641b31f971dcc7b87a76ea2916f1cb878"
MARKER = "generals_spatial_muon_dense_orientation=canonical_v1"
SCALE_STATEMENT = b"        float scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));"


def canonical_dense_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != INSTALLED_ALGO_SHA256:
        raise ValueError("Expected the pinned installed Fabric optimizer source")
    if source.count(SCALE_STATEMENT) != 1:
        raise ValueError("Expected exactly one upstream Muon aspect-ratio scale")
    replacement = (
        '        static const char metta_dense_orientation[] __attribute__((used)) = "'
        + MARKER
        + '";\n'
        "        bool metta_input_major = (R == 14112 && C == 32) || (R == 32 && C == 3530);\n"
        "        float scale = sqrtf(fmaxf(1.0f, metta_input_major\n"
        "            ? (float)C / (float)R : (float)R / (float)C));"
    ).encode()
    result = source.replace(SCALE_STATEMENT, replacement)
    if hashlib.sha256(result).hexdigest() != CANONICAL_ALGO_SHA256:
        raise ValueError("Canonical optimizer patch differs from its verified source")
    return result


def validate_geometry(config):
    raw = config.model_dump() if hasattr(config, "model_dump") else config
    from integrations.native_spatial_asset import validate_fabric
    validate_fabric(raw)
    if any(raw.get(key) for key in (
        "teacher", "losses", "horde", "rnd", "routing", "self_distillation",
        "ema_prior", "group_returns", "quantile_critic", "retrace",
    )):
        raise ValueError("Dense Muon orientation requires the plain PPO spatial model")


def install_build_hook(puffer_module):
    """Patch only the new, exclusive build after its pinned Fabric seam installs."""
    original = puffer_module.install_fabric
    if getattr(original, "_spatial_muon_orientation", False):
        raise RuntimeError("Dense Muon orientation build hook is already installed")

    @functools.wraps(original)
    def install(source, config):
        validate_geometry(config)
        words = original(source, config)
        path = Path(source) / "src/algo.cu"
        before = path.read_bytes()
        after = canonical_dense_source(before)
        path.write_bytes(after)
        receipt = dict(
            mode="canonical_dense", marker=MARKER,
            original_algo_sha256=hashlib.sha256(before).hexdigest(),
            patched_algo_sha256=hashlib.sha256(after).hexdigest(),
            checkpoint_order_changed=False, matrix_direction_changed=False,
            scales=[dict(name="global", storage_shape=[14112, 32],
                         logical_shape=[32, 14112], old=21.0, new=1.0),
                    dict(name="readout", storage_shape=[32, 3530],
                         logical_shape=[3530, 32], old=1.0, new=(3530 / 32) ** .5)],
            context_convolution="unchanged vector registration",
        )
        (Path(source).parent / "spatial-muon-orientation.json").write_text(
            json.dumps(receipt, indent=2) + "\n")
        print("SPATIAL_MUON_ORIENTATION " + json.dumps(receipt), flush=True)
        return words

    install._spatial_muon_orientation = True
    puffer_module.install_fabric = install


FINAL_STATELESS_ALGO_SHA256 = {
    1.01: "fe6f832ae684ecd99caea1b5b4a57057ba0d1627da6722401a3d0b124f3009f2",
    2.01: "3b87ff67e8871cee8fd515ba6f62a317d96e4f0066b39d78f00c03b18fdd7fe7",
}


def finalize_build_receipt(source, config):
    """Seal the compiled source after the stateless ABI transformation."""
    from integrations.spatial_muon_context import config_radius, geometry, geometry_receipt
    source = Path(source)
    radius = config_radius(config)
    build = source.parent
    receipt_path = build / "spatial-muon-orientation.json"
    receipt = json.loads(receipt_path.read_text())
    intermediate = geometry(radius)[3]
    final = hashlib.sha256((source / "src/algo.cu").read_bytes()).hexdigest()
    if (receipt["mode"] != "canonical_dense" or receipt["marker"] != MARKER
            or receipt["original_algo_sha256"] != INSTALLED_ALGO_SHA256
            or receipt["dense_algo_sha256"] != CANONICAL_ALGO_SHA256
            or receipt["patched_algo_sha256"] != intermediate
            or receipt.get("context_matrix") is not True
            or json.loads((build / "spatial-muon-context.json").read_text()) != geometry_receipt(radius)
            or final != FINAL_STATELESS_ALGO_SHA256[radius]):
        raise ValueError("Final spatial optimizer transformation chain differs")
    receipt.update(optimizer_algo_sha256=intermediate, patched_algo_sha256=final,
                   stateless_installer_sha256=hashlib.sha256(Path(__file__).with_name("puffer_stateless_spatial.py").read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def validate_build_mode(build, mode, *, context_matrix=False):
    if mode not in ("storage", "canonical"):
        raise ValueError("Spatial Muon orientation must be storage or canonical")
    if context_matrix and mode != "canonical":
        raise ValueError("Convolution matrix mode requires canonical dense scaling")
    build = Path(build)
    from integrations.puffer_rollout_memory import validate as validate_rollout_memory
    validate_rollout_memory(build)
    from integrations.spatial_muon_context import config_radius, geometry, validate_context_build
    validate_context_build(build, context_matrix)
    marked = MARKER.encode() in (build / "puffer").read_bytes()
    if marked != (mode == "canonical"):
        raise ValueError("Requested Muon orientation does not match the compiled executable")
    if not marked:
        return
    manifest = json.loads((build / "build.json").read_text())
    validate_geometry(manifest["config"]["fabric"])
    receipt = json.loads((build / "spatial-muon-orientation.json").read_text())
    source_hash = hashlib.sha256((build / "source/src/algo.cu").read_bytes()).hexdigest()
    if not context_matrix:
        raise ValueError("Current stateless spatial builds require convolution matrix mode")
    radius = config_radius(manifest["config"]["fabric"])
    expected_hash = FINAL_STATELESS_ALGO_SHA256[radius]
    if (receipt["mode"] != "canonical_dense" or receipt["marker"] != MARKER
            or receipt["original_algo_sha256"] != INSTALLED_ALGO_SHA256
            or receipt.get("optimizer_algo_sha256") != geometry(radius)[3]
            or receipt.get("stateless_installer_sha256") != hashlib.sha256(Path(__file__).with_name("puffer_stateless_spatial.py").read_bytes()).hexdigest()
            or receipt["patched_algo_sha256"] != source_hash
            or source_hash != expected_hash
            or bool(receipt.get("context_matrix", False)) != context_matrix
            or (context_matrix and receipt.get("dense_algo_sha256") != CANONICAL_ALGO_SHA256)):
        raise ValueError("Canonical Muon source and build receipt differ")


def install_runtime_guard(puffer_module, mode, *, context_matrix=False):
    original = puffer_module.prepare_run

    @functools.wraps(original)
    def prepare(build, output, config, *, name=None):
        validate_build_mode(build, mode, context_matrix=context_matrix)
        return original(build, output, config, name=name)

    puffer_module.prepare_run = prepare

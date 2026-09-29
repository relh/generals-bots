"""Muon convolution matrix updates with flat checkpoint/momentum order preserved."""

import functools
import hashlib
import json
from pathlib import Path

import numpy as np

from integrations.spatial_muon_orientation import CANONICAL_ALGO_SHA256, validate_geometry


GATHER_SHA256 = "695735db9987c62beb0ed534c6f0c2f7b52c881151175985536e279a74033a04"
CONTEXT_ALGO_SHA256 = "26b33054496feb99883901197d5894e082daf13e71849f59bae91dc96a81b592"
MARKER = "generals_spatial_muon_context=matrix_v1"
OFFSET = 564952
SHAPE = (32, 160)


def load_gather():
    raw = json.loads(Path(__file__).with_name("spatial_context_muon_gather.json").read_text())
    values = raw["gather"]
    if (raw["shape"] != list(SHAPE) or raw["offset"] != OFFSET
            or raw["gather_sha256"] != GATHER_SHA256
            or len(values) != 5120 or any(type(v) is not int for v in values)):
        raise ValueError("Convolution gather metadata differs from the pinned geometry")
    gather = np.asarray(values, dtype="<i4")
    if (not np.array_equal(np.sort(gather), np.arange(5120))
            or hashlib.sha256(gather.tobytes()).hexdigest() != GATHER_SHA256):
        raise ValueError("Convolution gather differs from the verified bijection")
    return gather


def validate_model_gather(model):
    kernel = np.asarray(model.context_kernel)
    if kernel.shape != (3, 3, 32, 32):
        raise ValueError("Convolution requires the verified F32 cross stencil")
    coordinates = np.argwhere(kernel >= 0)
    if len(coordinates) != 5120:
        raise ValueError("Convolution stencil does not contain exactly five neighbors")
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0],
                       coordinates[:, 2], coordinates[:, 3]))
    indices = kernel[tuple(coordinates[order].T)]
    gather = load_gather()
    if not np.array_equal(indices, OFFSET + gather):
        raise ValueError("Actual convolution sharing order differs from the pinned gather")
    return gather


def _patch_context_source(source):
    gather = load_gather()
    declaration = (
        'static const char metta_context_orientation[] __attribute__((used)) = "'
        + MARKER + '";\n'
        "__device__ __constant__ int metta_context_muon_gather[5120] = {\n"
        + ",\n".join(",".join(map(str, gather[i:i + 32])) for i in range(0, 5120, 32))
        + "\n};\n"
        "__global__ void muon_context_reorder(precision_t* __restrict__ dst,\n"
        "        const precision_t* __restrict__ src, bool scatter, int n) {\n"
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "    if (idx < n) {\n"
        "        int physical = metta_context_muon_gather[idx];\n"
        "        if (scatter) dst[physical] = src[idx];\n"
        "        else dst[idx] = src[physical];\n"
        "    }\n"
        "}\n\n"
    ).encode()
    replacements = [
        (b"// dst = scale * src  (write NS result + aspect scale into flat grad buffer)",
         declaration + b"// dst = scale * src  (write NS result + aspect scale into flat grad buffer)"),
        (b"        long R = e.shape[0], C = ne / R;",
         b"        long R = e.shape[0], C = ne / R;\n"
         b"        bool metta_context_matrix = R == 32 && C == 160 && offset - ne == 564952;"),
        (b"        int nblk = min((int)grid_size(ne), 256);",
         b"        if (metta_context_matrix) {\n"
         b"            muon_context_reorder<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(\n"
         b"                x_buf.data, x.data, false, (int)ne);\n"
         b"            puf_copy(&x, &x_buf, stream);\n"
         b"        }\n\n"
         b"        int nblk = min((int)grid_size(ne), 256);"),
        (b"        muon_store_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(\n"
         b"            gc_ptr, x_buf.data, scale, (int)ne);",
         b"        if (metta_context_matrix) {\n"
         b"            muon_context_reorder<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(\n"
         b"                gc_ptr, x_buf.data, true, (int)ne);\n"
         b"        } else {\n"
         b"            muon_store_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(\n"
         b"                gc_ptr, x_buf.data, scale, (int)ne);\n"
         b"        }"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Convolution optimizer patch seam differs from pinned source")
        source = source.replace(old, new)
    return source


def context_source(source):
    if hashlib.sha256(source).hexdigest() != CANONICAL_ALGO_SHA256:
        raise ValueError("Convolution requires the pinned canonical dense optimizer")
    patched = _patch_context_source(source)
    if hashlib.sha256(patched).hexdigest() != CONTEXT_ALGO_SHA256:
        raise ValueError("Convolution optimizer source differs from its verified patch")
    return patched


def install_build_hook(puffer_module):
    original = puffer_module.install_fabric
    if getattr(original, "_spatial_muon_context", False):
        raise RuntimeError("Convolution Muon build hook is already installed")

    @functools.wraps(original)
    def install(source, config):
        validate_geometry(config)
        words = original(source, config)
        path = Path(source) / "src/algo.cu"
        patched = context_source(path.read_bytes())
        path.write_bytes(patched)
        build = Path(source).parent
        receipt = dict(marker=MARKER, base_algo_sha256=CANONICAL_ALGO_SHA256,
                       patched_algo_sha256=CONTEXT_ALGO_SHA256,
                       gather_sha256=GATHER_SHA256, offset=OFFSET, shape=list(SHAPE),
                       checkpoint_order_changed=False, momentum_order_changed=False,
                       reorders_gradient_scratch=True)
        (build / "spatial-muon-context.json").write_text(json.dumps(receipt, indent=2) + "\n")
        dense_path = build / "spatial-muon-orientation.json"
        dense = json.loads(dense_path.read_text())
        dense.update(dense_algo_sha256=CANONICAL_ALGO_SHA256,
                     patched_algo_sha256=CONTEXT_ALGO_SHA256, context_matrix=True)
        dense_path.write_text(json.dumps(dense, indent=2) + "\n")
        print("SPATIAL_MUON_CONTEXT " + json.dumps(receipt), flush=True)
        return words

    install._spatial_muon_context = True
    puffer_module.install_fabric = install


def validate_context_build(build, enabled):
    build = Path(build)
    marked = MARKER.encode() in (build / "puffer").read_bytes()
    if marked != enabled:
        raise ValueError("Convolution matrix mode does not match compiled executable")
    if not enabled:
        return
    load_gather()
    receipt = json.loads((build / "spatial-muon-context.json").read_text())
    expected = dict(marker=MARKER, base_algo_sha256=CANONICAL_ALGO_SHA256,
                    patched_algo_sha256=CONTEXT_ALGO_SHA256, gather_sha256=GATHER_SHA256,
                    offset=OFFSET, shape=list(SHAPE), checkpoint_order_changed=False,
                    momentum_order_changed=False, reorders_gradient_scratch=True)
    if receipt != expected:
        raise ValueError("Convolution optimizer receipt differs from verified mapping")

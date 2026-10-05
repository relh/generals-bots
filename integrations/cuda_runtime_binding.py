"""Bind the portable pilot to its installed CUDA wheels before interpreter startup."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
import sysconfig


def library_path(existing: str, platlib: Path) -> str:
    required = [platlib / "nvidia" / name / "lib" for name in ("cu13", "cudnn", "nccl")]
    if not all(path.is_dir() for path in required):
        raise RuntimeError("Portable CUDA wheel library directories are missing")
    return os.pathsep.join(dict.fromkeys([*map(str, required), *filter(None, existing.split(os.pathsep))]))


def configure() -> None:
    """Re-exec: changing LD_LIBRARY_PATH after dlopen startup is insufficient."""
    target = library_path(os.environ.get("LD_LIBRARY_PATH", ""), Path(sysconfig.get_path("platlib")))
    if os.environ.get("LD_LIBRARY_PATH") != target:
        env = dict(os.environ, LD_LIBRARY_PATH=target)
        os.execvpe(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)
    os.environ["GENERALS_AUDIT_CUDA_RUNTIME"] = "1"


def audit() -> dict:
    library = ctypes.CDLL("libcublas.so.13")
    query = library.cublasGetProperty
    query.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    query.restype = ctypes.c_int
    version = []
    for property_id in range(3):
        value = ctypes.c_int()
        if query(property_id, ctypes.byref(value)):
            raise RuntimeError("Could not query loaded cuBLAS version")
        version.append(value.value)
    if not (version[0] == 13 and version[1] >= 2):
        raise RuntimeError(f"Unsafe cuBLAS runtime binding: {version}")
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/libcublas" in line})
    root = Path(sysconfig.get_path("platlib")) / "nvidia/cu13/lib"
    if not paths or any(not Path(path).is_relative_to(root) for path in paths):
        raise RuntimeError("cuBLAS loaded outside the audited CUDA wheel directory")
    result = dict(cublas_version=version, loaded_paths=paths, pid=os.getpid())
    print("CUDA_RUNTIME_BINDING " + json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    configure()
    audit()

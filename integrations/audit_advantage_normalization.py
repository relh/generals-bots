"""Verify the built FP32 normalizer on CUDA; no training or throughput claim."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures(horizon):
    values = ((np.arange(8192) * 37 % 1021) / 128 - 2).astype(np.float32)
    for name, data in (("varying", values), ("constant", np.full(8192, 3, np.float32))):
        data.reshape(-1, horizon)[:, -1] = 1e20
        yield name, data


def reference(values, horizon):
    active = values.reshape(-1, horizon)[:, :-1].astype(np.float64)
    expected = np.zeros(values.size, np.float64).reshape(-1, horizon)
    expected[:, :-1] = (active - active.mean()) / (active.std(ddof=1) + 1e-8)
    return expected.ravel()


def compiler_command(header, source, executable):
    return ["nvcc", "-std=c++17", "-O2", "-DPRECISION_FLOAT", "-I", str(header.parent),
            str(source), "-o", str(executable)]


def cuda_source(native):
    # Compile the actual trainer's precision declaration, rather than assuming
    # a harness alias establishes the build's dtype.
    start = native.index("#ifdef PRECISION_FLOAT\n")
    precision = native[start:native.index("#endif", start) + len("#endif")]
    if "typedef float precision_t;" not in precision or "constexpr bool USE_BF16 = false;" not in precision:
        raise ValueError("Native precision declaration changed")
    return '''#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
''' + precision + r'''
#include "metta_advantage.cuh"
static_assert(sizeof(precision_t) == 4 && !USE_BF16);
#define OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);} } while(0)
int main(int argc, char** argv) {
  if(argc!=4) return 3;
  int horizon=atoi(argv[3]), count=8192;
  if(horizon!=128 && horizon!=256) return 4;
  std::vector<float> host(count);
  FILE* input=fopen(argv[1],"rb"); if(!input) return 5;
  if(fread(host.data(),sizeof(float),count,input)!=count) return 6;
  fclose(input);
  float* device; OK(cudaMalloc(&device,count*sizeof(float)));
  cudaStream_t stream; OK(cudaStreamCreate(&stream));
  OK(cudaMemcpyAsync(device,host.data(),count*sizeof(float),cudaMemcpyHostToDevice,stream));
  metta_standardize_ppo_advantages<<<1,256,0,stream>>>(device,count,horizon);
  OK(cudaGetLastError());
  OK(cudaMemcpyAsync(host.data(),device,count*sizeof(float),cudaMemcpyDeviceToHost,stream));
  OK(cudaStreamSynchronize(stream));
  FILE* output=fopen(argv[2],"wb"); if(!output) return 7;
  if(fwrite(host.data(),sizeof(float),count,output)!=count) return 8;
  if(fclose(output)) return 9;
  OK(cudaStreamDestroy(stream)); OK(cudaFree(device));
}
'''


def audit(build, output):
    metadata = json.loads((build / "build.json").read_text())
    if metadata["config"]["precision"] != "float32" or digest(build / "puffer") != metadata["binary_sha256"]:
        raise ValueError("Expected identified FP32 native build")
    header = build / "source/src/metta_advantage.cuh"
    if header.read_bytes() != Path(__file__).with_name("puffer_advantage_normalization.cuh").read_bytes():
        raise ValueError("Built normalizer differs from production header")
    native = build / "source/src/pufferl.cu"
    if native.read_text().count("graph.mb_advantages.data, count, Tmb);") != 1:
        raise ValueError("Trainer does not pass minibatch horizon")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix="advantage-cuda-") as temporary:
        directory = Path(temporary)
        source, executable = directory / "audit.cu", directory / "audit"
        source.write_text(cuda_source(native.read_text()))
        command = compiler_command(header, source, executable)
        subprocess.run(command, check=True, timeout=120)
        for horizon in (128, 256):
            for name, values in fixtures(horizon):
                prefix = output / f"h{horizon}-{name}"
                values.astype("<f4").tofile(str(prefix) + "-input.bin")
                subprocess.run([str(executable), str(prefix) + "-input.bin", str(prefix) + "-output.bin",
                                str(horizon)], check=True, timeout=30)
                actual = np.fromfile(str(prefix) + "-output.bin", dtype="<f4")
                expected = reference(values, horizon)
                # Inputs have bounded, well-conditioned variance (or exactly
                # representable constant values). 1e-5 covers FP32 reduction
                # rounding, while a poisoned bootstrap inclusion fails hugely.
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
                assert np.isfinite(actual).all()
                np.testing.assert_array_equal(actual.reshape(-1, horizon)[:, -1], 0)
                cases.append(dict(horizon=horizon, count=8192, threads=256, fixture=name,
                                  max_abs_error=float(np.max(np.abs(actual - expected))),
                                  input_sha256=digest(str(prefix) + "-input.bin"),
                                  output_sha256=digest(str(prefix) + "-output.bin")))
    report = dict(schema="generals-advantage-normalization-cuda-v1", passed=True, backend="cuda",
                  build_sha256=digest(build / "build.json"), binary_sha256=metadata["binary_sha256"],
                  header_sha256=digest(header), generated_pufferl_sha256=digest(native),
                  audit_module_sha256=digest(__file__), compiler_flags=command[1:4],
                  rtol=1e-5, atol=1e-5, cases=cases,
                  scope="Actual built FP32 header and 256-thread CUDA reduction; synthetic inputs only")
    (output / "admission.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.build.resolve(), args.output.resolve()), allow_nan=False))


if __name__ == "__main__":
    main()

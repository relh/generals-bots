"""Check the generated trainer's actual CUDA normalizer against NumPy."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    header = (args.build / 'source/src/metta_advantage.cuh').resolve()
    trainer = (args.build / 'source/src/pufferl.cu').read_text()
    assert trainer.count('metta_standardize_ppo_advantages<<<1, 256, 0, stream>>>') == 1
    call = trainer.index('metta_standardize_ppo_advantages<<<1, 256, 0, stream>>>')
    assert trainer.index('metta_retrace_advantages(pufferl->train_activs, graph,') < call
    assert call < trainer.index('ppo_loss_fwd_bwd(dec, p_logstd, graph,')
    assert 'norm_adv = 0' in (args.build / 'source/config/default.ini').read_text()
    args.output.mkdir(parents=True, exist_ok=False)
    program = args.output / 'audit.cu'
    assert '"' not in str(header)
    program.write_text(r'''#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
typedef float precision_t;
__device__ float to_float(float value) { return value; }
__device__ float from_float(float value) { return value; }
#define CHECK(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(err)); exit(2); } } while (0)
''' + '#include "' + str(header) + '"\n' + r'''
int main(int argc, char** argv) {
    if (argc != 4) return 1;
    int count = atoi(argv[3]);
    float* host = (float*)malloc(count * sizeof(float));
    FILE* input = fopen(argv[1], "rb");
    if (!input || fread(host, sizeof(float), count, input) != (size_t)count) return 1;
    fclose(input);
    float* device;
    CHECK(cudaMalloc(&device, count * sizeof(float)));
    CHECK(cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice));
    metta_standardize_ppo_advantages<<<1, 256>>>(device, count);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* output = fopen(argv[2], "wb");
    if (!output || fwrite(host, sizeof(float), count, output) != (size_t)count) return 1;
    fclose(output);
    CHECK(cudaFree(device));
    free(host);
    return 0;
}
''')
    binary = args.output / 'audit'
    subprocess.run(['nvcc', '-O2', '-arch=sm_100', str(program), '-o', str(binary)], check=True)
    cases = {
        'two': np.array([-2, 5], np.float32),
        'zero': np.zeros(32768, np.float32),
        'constant': np.full(32768, 3, np.float32),
        'dense': np.linspace(-0.05, 0.06, 32768, dtype=np.float32),
        'small': np.linspace(-1e-8, 1e-8, 32768, dtype=np.float32),
        'sparse': np.concatenate([np.ones(1, np.float32), np.zeros(32767, np.float32)]),
    }
    report = {'header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(), 'cases': {}}
    for name, values in cases.items():
        input_path, output_path = args.output / (name + '.input'), args.output / (name + '.output')
        values.tofile(input_path)
        subprocess.run([str(binary), str(input_path), str(output_path), str(values.size)], check=True)
        actual = np.fromfile(output_path, dtype=np.float32)
        reference = (values.astype(np.float64) - values.astype(np.float64).mean()) / (values.astype(np.float64).std(ddof=1) + 1e-8)
        np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
        report['cases'][name] = {'count': values.size, 'max_absolute_error': float(np.max(np.abs(actual-reference))), 'mean': float(actual.mean()), 'sample_std': float(actual.std(ddof=1))}
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ADVANTAGE_NORMALIZATION_PARITY_OK', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

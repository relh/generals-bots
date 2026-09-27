"""Compare frozen JAX inference with the actual pinned CUDA arch_forward."""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import numpy as np

from integrations.native_puffer_policy import NativePufferPolicy

HARNESS = r'''
void read_file(const char* name, void* data, size_t bytes) {
    FILE* f = fopen(name, "rb"); assert(f);
    assert(fread(data, 1, bytes, f) == bytes); fclose(f);
}
int main(int argc, char** argv) {
    assert(argc == 5);
    const int B=4, H=128, L=4, T=6, O=6174, D=1768;
    cublas_init_handle();
    Arch arch = build_arch(O, H, L, D-1, false, 16);
    Allocator params={}, acts={};
    Weights weights = weights_create(&arch, &params);
    Activations activations = arch_reg_rollout(&arch, weights, &acts, B);
    Prec input={.shape={B,O}}, state={.shape={L,B,H}};
    alloc_register(&acts, &input); alloc_register(&acts, &state);
    alloc_create(&params); alloc_create(&acts);
    void* parameters=malloc(params.total_bytes);
    read_file(argv[1], parameters, params.total_bytes);
    cudaMemcpy(params.mem, parameters, params.total_bytes, cudaMemcpyHostToDevice);
    float* observations=(float*)malloc(T*B*O*sizeof(float));
    float* host_state=(float*)malloc(L*B*H*sizeof(float));
    float* host_output=(float*)malloc(B*D*sizeof(float));
    read_file(argv[2], observations, T*B*O*sizeof(float));
    read_file(argv[3], host_state, L*B*H*sizeof(float));
    cudaMemcpy(state.data,host_state,L*B*H*sizeof(float),cudaMemcpyHostToDevice);
    FILE* output=fopen(argv[4],"wb"); assert(output);
    for(int t=0;t<T;t++) {
        if(t==3) {
            // Reset just one seat: retain the other seats' recurrent state.
            cudaMemcpy(host_state,state.data,L*B*H*sizeof(float),cudaMemcpyDeviceToHost);
            for(int l=0;l<L;l++) memset(host_state+(l*B+1)*H,0,H*sizeof(float));
            cudaMemcpy(state.data,host_state,L*B*H*sizeof(float),cudaMemcpyHostToDevice);
        }
        cudaMemcpy(input.data,observations+t*B*O,B*O*sizeof(float),cudaMemcpyHostToDevice);
        Prec decoded=arch_forward(&arch,weights,activations,input,state,0);
        assert(cudaDeviceSynchronize()==cudaSuccess);
        cudaMemcpy(host_output,decoded.data,B*D*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(host_state,state.data,L*B*H*sizeof(float),cudaMemcpyDeviceToHost);
        assert(fwrite(host_output,sizeof(float),B*D,output)==B*D);
        assert(fwrite(host_state,sizeof(float),L*B*H,output)==L*B*H);
    }
    fclose(output);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=False)
    policy = NativePufferPolicy(args.build, args.training, args.checkpoint, args.sha256)
    assert (policy.hidden, policy.layers) == (128, 4)
    source = args.source / "src"
    prefix = (source / "pufferl.cu").read_text().split('#include "protein.cu"')[0]
    for line in ('#include "ini.h"', '#include "metta_sweep.cuh"', '#include ENV_HEADER',
                 '#include <nccl.h>', '#include <nvml.h>', '#include <nvtx3/nvToolsExt.h>'):
        prefix = prefix.replace(line, "")
    harness = args.output / "native_forward.cu"
    harness.write_text("#define NUM_ATNS 2\n#define ACT_SIZES {1765,2}\n" + prefix + HARNESS)
    executable = args.output / "native_forward"
    subprocess.run(["nvcc", "-O2", "-arch=sm_100", "-std=c++17", "-DPRECISION_FLOAT",
                    "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361", "-I" + str(source),
                    str(harness), "-lcublas", "-lcurand", "-o", str(executable)], check=True)
    rng = np.random.default_rng(20260927)
    observations = rng.normal(size=(6, 4, 6174)).astype(np.float32) / 8
    states = rng.uniform(size=(4, 4, 128)).astype(np.float32)
    obs_path, state_path, output_path = [args.output / name for name in ("obs.bin", "state.bin", "native.bin")]
    observations.tofile(obs_path)
    states.tofile(state_path)
    subprocess.run([str(executable), str(args.checkpoint), str(obs_path), str(state_path),
                    str(output_path)], check=True)
    native = np.fromfile(output_path, dtype=np.float32).reshape(6, 4 * 1768 + 4 * 4 * 128)
    state = jax.numpy.asarray(states)
    max_logits, max_state = 0.0, 0.0
    for turn in range(6):
        if turn == 3:
            state = state.at[:, 1].set(0)
        before = state
        decoded, state = policy.forward(jax.numpy.asarray(observations[turn]), before)
        expected_logits = native[turn, :4 * 1768].reshape(4, 1768)
        expected_state = native[turn, 4 * 1768:].reshape(4, 4, 128)
        np.testing.assert_allclose(decoded, expected_logits, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(state, expected_state, atol=2e-5, rtol=2e-5)
        max_logits = max(max_logits, float(np.max(np.abs(np.asarray(decoded) - expected_logits))))
        max_state = max(max_state, float(np.max(np.abs(np.asarray(state) - expected_state))))
        masks = rng.random((4, 1767)) > .5
        masks[:, 1764:] = True
        reference = np.where(masks, expected_logits[:, :1767], -np.inf)
        expected_actions = np.stack((reference[:, :1765].argmax(-1), reference[:, 1765:].argmax(-1)), -1)
        actual, action_state = policy.actions(observations[turn], masks, before)
        np.testing.assert_array_equal(actual, expected_actions)
        np.testing.assert_allclose(action_state, expected_state, atol=2e-5, rtol=2e-5)
    result = dict(checkpoint_sha256=args.sha256, recurrent_steps=24, partial_seat_reset=True,
                  max_logit_difference=max_logits, max_state_difference=max_state,
                  scope="CUDA arch_forward vs frozen JAX logits, values, recurrent states and masked argmax")
    (args.output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

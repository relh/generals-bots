"""Check default native MinGRU training gradients against JAX autodiff on GPU."""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from integrations.native_puffer_policy import NativePufferPolicy

HARNESS = r'''
void read_file(const char* name, void* data, size_t bytes) {
    FILE* f=fopen(name,"rb"); assert(f);
    assert(fread(data,1,bytes,f)==bytes); fclose(f);
}
void write_file(const char* name, void* data, size_t bytes) {
    FILE* f=fopen(name,"wb"); assert(f);
    assert(fwrite(data,1,bytes,f)==bytes); fclose(f);
}
int main(int argc,char** argv) {
    assert(argc==9);
    const int B=4,T=16,H=128,L=4,O=6174,D=1768,A=6,OFF=1;
    cublas_init_handle();
    Arch arch=build_arch(O,H,L,D-1,false,T);
    Allocator params={},acts={},grads={};
    Weights weights=weights_create(&arch,&params);
    Activations activations=arch_reg_train(&arch,weights,&acts,&grads,B*T);
    Prec input={.shape={B,T,O}},state={.shape={L,A,H}},terminals={.shape={B,T}};
    Float actor={.shape={B,T,D-1}},critic={.shape={B,T,1}};
    alloc_register(&acts,&input); alloc_register(&acts,&state); alloc_register(&acts,&terminals);
    alloc_register(&acts,&actor); alloc_register(&acts,&critic);
    alloc_create(&params); alloc_create(&acts); alloc_create(&grads);
    assert(params.total_bytes==grads.total_bytes);
    void* host=malloc(params.total_bytes);
    read_file(argv[1],host,params.total_bytes);
    cudaMemcpy(params.mem,host,params.total_bytes,cudaMemcpyHostToDevice);
    const char* names[]={argv[2],argv[3],argv[4],argv[5],argv[6]};
    void* pointers[]={input.data,state.data,terminals.data,actor.data,critic.data};
    size_t sizes[]={B*T*O*4,L*A*H*4,B*T*4,B*T*(D-1)*4,B*T*4};
    for(int i=0;i<5;i++) {
        void* buf=malloc(sizes[i]);read_file(names[i],buf,sizes[i]);
        cudaMemcpy(pointers[i],buf,sizes[i],cudaMemcpyHostToDevice);free(buf);
    }
    // Same model operations as arch_forward_train, excluding PPO's auxiliary cache kernel.
    Prec x=arch.encoder.forward(weights.encoder,activations.encoder,*puf_squeeze(&input,0),0);
    x=arch.network.forward_train(weights.network,*puf_unsqueeze(&x,0,B,T),
        state,terminals,activations.network,OFF,0);
    Prec out=arch.decoder.forward(weights.decoder,activations.decoder,*puf_squeeze(&x,0),0);
    float* decoded=(float*)malloc(B*T*D*4);
    cudaMemcpy(decoded,out.data,B*T*D*4,cudaMemcpyDeviceToHost);
    write_file(argv[7],decoded,B*T*D*4);
    Float logstd={};
    arch_backward(&arch,weights,activations,actor,logstd,critic,0);
    assert(cudaDeviceSynchronize()==cudaSuccess);
    cudaMemcpy(host,grads.mem,grads.total_bytes,cudaMemcpyDeviceToHost);
    write_file(argv[8],host,grads.total_bytes);
    MinGRUActivations* network=(MinGRUActivations*)activations.network;
    float* states=(float*)malloc(2*L*B*H*4);
    for(int l=0;l<L;l++) {
        cudaMemcpy(states+l*B*H,network->scan_bufs[l].next_state.data,B*H*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(states+(L+l)*B*H,network->scan_bufs[l].grad_state.data,B*H*4,cudaMemcpyDeviceToHost);
    }
    char name[4096];snprintf(name,sizeof(name),"%s.states",argv[8]);
    write_file(name,states,2*L*B*H*4);
}
'''


def sequence(parameters, state, observations, terminals):
    encoder, decoder, recurrent = parameters

    def step(previous, frame):
        values, done = frame
        previous = jnp.where(done[None, :, None] != 0, 0, previous)
        x = jnp.matmul(values, encoder.T, precision=jax.lax.Precision.HIGHEST)
        updated = []
        for layer, weight in enumerate(recurrent):
            combined = jnp.matmul(x, weight.T, precision=jax.lax.Precision.HIGHEST)
            hidden, gate, projection = jnp.split(combined, 3, -1)
            z = jax.nn.sigmoid(gate)
            candidate = jnp.where(hidden >= 0, hidden + .5, jax.nn.sigmoid(hidden))
            delta = candidate - previous[layer]
            h = jnp.where(z < .5, previous[layer] + z * delta, candidate - (1 - z) * delta)
            s = jax.nn.sigmoid(projection)
            x = s * h + (1 - s) * x
            updated.append(h)
        return jnp.stack(updated), jnp.matmul(x, decoder.T, precision=jax.lax.Precision.HIGHEST)

    final, decoded = jax.lax.scan(step, state, (observations.swapaxes(0, 1), terminals.T))
    return decoded.swapaxes(0, 1), final


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "build", "training", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sha256", required=True)
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
    harness, executable = args.output / "backward.cu", args.output / "backward"
    harness.write_text("#define NUM_ATNS 2\n#define ACT_SIZES {1765,2}\n" + prefix + HARNESS)
    subprocess.run(["nvcc", "-O2", "-arch=sm_100", "-std=c++17", "-DPRECISION_FLOAT",
                    "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361", "-I" + str(source),
                    str(harness), "-lcublas", "-lcurand", "-o", str(executable)], check=True)
    rng = np.random.default_rng(20260928)
    observations = rng.normal(size=(4, 16, 6174)).astype(np.float32) / 8
    initial = rng.uniform(size=(4, 6, 128)).astype(np.float32)
    terminals = np.zeros((4, 16), np.float32)
    terminals[0, 0], terminals[1, 7], terminals[3, 11] = 1, 1, 1
    cotangents = rng.normal(size=(4, 16, 1768)).astype(np.float32) / 100
    paths = [args.output / name for name in ("obs.bin", "state.bin", "done.bin", "actor.bin", "critic.bin")]
    for path, array in zip(paths, (observations, initial, terminals, cotangents[:, :, :-1],
                                  cotangents[:, :, -1:]), strict=True):
        array.tofile(path)
    native_output, native_gradient = args.output / "decoded.bin", args.output / "gradient.bin"
    subprocess.run([str(executable), str(args.checkpoint), *map(str, paths),
                    str(native_output), str(native_gradient)], check=True)
    parameters = policy.encoder, policy.decoder, policy.recurrent

    def loss(weights, state):
        decoded, final = sequence(weights, state, jnp.asarray(observations), jnp.asarray(terminals))
        return jnp.sum(decoded * jnp.asarray(cotangents)), (decoded, final)

    (_, (decoded, final)), (gradient, state_gradient) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    )(parameters, jnp.asarray(initial[:, 1:5]))
    expected = np.fromfile(native_output, np.float32).reshape(4, 16, 1768)
    states = np.fromfile(str(native_gradient) + ".states", np.float32).reshape(2, 4, 4, 128)
    np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(final, states[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(state_gradient, states[1], rtol=2e-5, atol=2e-5)
    expected_gradient = np.fromfile(native_gradient, np.float32)
    actual_gradient = np.concatenate([np.asarray(g).ravel() for g in jax.tree.leaves(gradient)])
    assert expected_gradient.size == actual_gradient.size == 1213184
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    result = dict(checkpoint_sha256=args.sha256, sequences=4, horizon=16, layers=4, hidden=128,
                  initial_state_agent_offset=1, terminal_resets=3,
                  max_forward_difference=float(np.max(np.abs(np.asarray(decoded) - expected))),
                  max_gradient_difference=float(np.max(np.abs(actual_gradient - expected_gradient))),
                  max_initial_state_gradient_difference=float(np.max(np.abs(np.asarray(state_gradient) - states[1]))),
                  scope="CUDA model forward/backward vs JAX autodiff; does not test PPO or Muon math")
    (args.output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

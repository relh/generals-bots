"""Compare the actual masked, two-head CUDA PPO loss with JAX autodiff."""

import argparse
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HARNESS = r'''
int main(int argc,char** argv) {
    assert(argc==3);
    const int B=4,T=4,D=1768,NT=B*T;
    Allocator alloc={};
    Prec logits={.shape={B,T,D}},empty={};
    Float losses={.shape={LOSS_N+1}};
    TrainGraph graph={};
    graph.mb_actions={.shape={B,T,2}};
    graph.mb_action_mask={.shape={B,T,D-1}};
    graph.mb_logprobs={.shape={B,T}};
    graph.mb_advantages={.shape={B,T}};
    graph.mb_values={.shape={B,T}};
    graph.mb_returns={.shape={B,T}};
    graph.mb_imp={.shape={B,T}};
    graph.mb_gae_v={.shape={B,T}};
    alloc_register(&alloc,&logits);alloc_register(&alloc,&losses);
    alloc_register(&alloc,&graph.mb_actions);alloc_register(&alloc,&graph.mb_action_mask);
    alloc_register(&alloc,&graph.mb_logprobs);alloc_register(&alloc,&graph.mb_advantages);
    alloc_register(&alloc,&graph.mb_values);alloc_register(&alloc,&graph.mb_returns);
    alloc_register(&alloc,&graph.mb_imp);alloc_register(&alloc,&graph.mb_gae_v);
    PPOBufs buffers={};register_ppo_buffers(buffers,&alloc,B,T,D-1,false);
    alloc_create(&alloc);
    FILE* input=fopen(argv[1],"rb");assert(input);
    void* pointers[]={logits.data,graph.mb_action_mask.data,graph.mb_actions.data,
        graph.mb_logprobs.data,graph.mb_advantages.data,graph.mb_values.data,graph.mb_returns.data};
    size_t counts[]={NT*D,NT*(D-1),NT*2,NT,NT,NT,NT};
    for(int i=0;i<7;i++) {
        float* host=(float*)malloc(counts[i]*4);
        assert(fread(host,4,counts[i],input)==counts[i]);
        cudaMemcpy(pointers[i],host,counts[i]*4,cudaMemcpyHostToDevice);free(host);
    }
    fclose(input);
    int sizes[]={1765,2};int* device_sizes;
    cudaMalloc(&device_sizes,sizeof(sizes));cudaMemcpy(device_sizes,sizes,sizeof(sizes),cudaMemcpyHostToDevice);
    FILE* output=fopen(argv[2],"wb");assert(output);
    float entropies[]={0.0f,0.001f};
    for(float entropy:entropies) {
        cudaMemset(losses.data,0,(LOSS_N+1)*4);
        cudaMemcpy(buffers.ent_coef,&entropy,4,cudaMemcpyHostToDevice);
        cache_imp_and_v<<<grid_size(NT),BLOCK_SIZE>>>(logits,graph.mb_actions.data,
            graph.mb_logprobs.data,graph.mb_action_mask.data,empty,device_sizes,
            graph.mb_imp.data,graph.mb_gae_v.data,buffers.grad_logits.data,buffers.grad_values.data);
        ppo_loss_fwd_bwd(logits,empty,graph,device_sizes,losses.data,.2f,.2f,1.0f,
            buffers.ent_coef,buffers,false,0);
        assert(cudaDeviceSynchronize()==cudaSuccess);
        void* arrays[]={buffers.grad_logits.data,buffers.grad_values.data,losses.data};
        size_t lengths[]={NT*(D-1),NT,LOSS_N+1};
        for(int i=0;i<3;i++) {
            float* host=(float*)malloc(lengths[i]*4);
            cudaMemcpy(host,arrays[i],lengths[i]*4,cudaMemcpyDeviceToHost);
            assert(fwrite(host,4,lengths[i],output)==lengths[i]);free(host);
        }
    }
    fclose(output);
}
'''


def statistics(decoded, masks, actions):
    entropy, logp = jnp.zeros(16), jnp.zeros(16)
    for head, (start, stop) in enumerate(((0, 1765), (1765, 1767))):
        logits = jnp.where(masks[:, start:stop], decoded[:, start:stop], -1e4)
        logps = jax.nn.log_softmax(logits, axis=-1)
        entropy -= jnp.sum(jnp.exp(logps) * logps, axis=-1)
        logp += logps[jnp.arange(16), actions[:, head]]
    return logp, entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.devices()[0].platform == "gpu"
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source / "src"
    prefix = (source / "pufferl.cu").read_text().split('#include "protein.cu"')[0]
    for line in ('#include "ini.h"', '#include "metta_sweep.cuh"', '#include ENV_HEADER',
                 '#include <nccl.h>', '#include <nvml.h>', '#include <nvtx3/nvToolsExt.h>'):
        prefix = prefix.replace(line, "")
    harness, executable = args.output / "ppo.cu", args.output / "ppo"
    harness.write_text("#define NUM_ATNS 2\n#define ACT_SIZES {1765,2}\n" + prefix + HARNESS)
    subprocess.run(["nvcc", "-O2", "-arch=sm_100", "-std=c++17", "-DPRECISION_FLOAT",
                    "-Xcompiler=-Wno-narrowing", "--diag-suppress=2361", "-I" + str(source),
                    str(harness), "-lcublas", "-lcurand", "-o", str(executable)], check=True)
    rng = np.random.default_rng(20260929)
    decoded = rng.normal(size=(16, 1768)).astype(np.float32) / 3
    masks = rng.random((16, 1767)) < .1
    masks[:, 1764:] = True
    actions = np.stack((np.array([rng.choice(np.flatnonzero(row[:1765])) for row in masks]),
                        rng.integers(0, 2, 16)), axis=-1).astype(np.int32)
    values = rng.uniform(-.8, .8, 16).astype(np.float32)
    decoded[:, -1] = values + np.tile([-.5, -.1, .1, .5], 4).astype(np.float32)
    returns = values + np.tile([-.6, .7, -.7, .6, .6, .7, -.7, -.6], 2).astype(np.float32)
    advantages = np.tile([1.7, -1.7, -.4, .4, -1.7, 1.7, .4, -.4], 2).astype(np.float32)
    logp, _ = statistics(jnp.asarray(decoded), jnp.asarray(masks), jnp.asarray(actions))
    old_logp = np.asarray(logp) - np.tile(np.log([1.5, .5, 1.1, .9]), 4).astype(np.float32)
    fixture, output = args.output / "fixture.bin", args.output / "native.bin"
    with fixture.open("wb") as stream:
        for array in (decoded, masks.astype(np.float32), actions.astype(np.float32),
                      old_logp, advantages, values, returns):
            array.tofile(stream)
    subprocess.run([str(executable), str(fixture), str(output)], check=True)
    native = np.fromfile(output, np.float32).reshape(2, 16 * 1768 + 9)
    reports = []
    for index, entropy_coefficient in enumerate((0.0, .001)):
        def loss(current):
            current_logp, entropy = statistics(current, jnp.asarray(masks), jnp.asarray(actions))
            logratio = current_logp - jnp.asarray(old_logp)
            ratio = jnp.exp(logratio)
            pg = jnp.maximum(-advantages * ratio, -advantages * jnp.clip(ratio, .8, 1.2))
            value_prediction = current[:, -1]
            clipped_value = values + jnp.clip(value_prediction - values, -.2, .2)
            vf = .5 * jnp.maximum((value_prediction - returns)**2, (clipped_value - returns)**2)
            total = jnp.mean(pg + vf - entropy_coefficient * entropy)
            metrics = jnp.array([pg.mean(), vf.mean(), entropy.mean(), total,
                                 -logratio.mean(), (ratio - 1 - logratio).mean(),
                                 (jnp.abs(ratio - 1) > .2).mean(), ratio.mean()])
            return total, metrics

        (_, metrics), gradient = jax.jit(jax.value_and_grad(loss, has_aux=True))(jnp.asarray(decoded))
        expected_actor = native[index, :16 * 1767].reshape(16, 1767)
        expected_value = native[index, 16 * 1767:16 * 1768]
        expected_metrics = native[index, -9:-1]
        np.testing.assert_allclose(gradient[:, :-1], expected_actor, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(gradient[:, -1], expected_value, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(metrics, expected_metrics, rtol=2e-5, atol=2e-5)
        assert native[index, -1] == 1
        assert (np.asarray(gradient[:, :-1])[~masks] == 0).all()
        assert (expected_actor[~masks] == 0).all()
        reports.append(dict(entropy_coefficient=entropy_coefficient,
                            max_actor_gradient_difference=float(np.max(np.abs(
                                np.asarray(gradient[:, :-1]) - expected_actor))),
                            max_value_gradient_difference=float(np.max(np.abs(
                                np.asarray(gradient[:, -1]) - expected_value))),
                            max_metric_difference=float(np.max(np.abs(np.asarray(metrics) - expected_metrics)))))
    result = dict(cases=16, heads=[1765, 2], policy_clip=.2, value_clip=.2, comparisons=reports,
                  scope="CUDA cache/PPO vs autodiff: masks, policy/value clipping, raw advantages and entropy")
    (args.output / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

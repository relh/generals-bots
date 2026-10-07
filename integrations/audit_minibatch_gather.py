"""GPU-only admission of the built gather kernel and its spatial PPO inputs."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CUDA = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
using precision_t = float;
#include "metta_rollout_memory.cuh"
#define OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);} } while(0)
__global__ void fill(float* src, long long count, bool mask) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count) src[i]=mask ? float((i*17+i/3529)%5!=0) : __uint_as_float(0x3e000000u | ((unsigned long long)i*2654435761u & 0x7fffffu));
}
__global__ void baseline(float* dst,const float* src,int C) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<(long long)4096*128*C){long long agent=i/(128*C),time=(i/C)%128,feature=i%C;dst[i]=src[(time*4096+agent)*C+feature];}
}
__global__ void equal_bits(const float* a,const float* b,long long count,unsigned int* errors) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count && __float_as_uint(a[i])!=__float_as_uint(b[i])) atomicAdd(errors,1u);
}
int main(int argc,char** argv){
 if(argc!=2)return 2;int devices=0;OK(cudaGetDeviceCount(&devices));if(devices!=1)return 3;
 cudaStream_t stream;OK(cudaStreamCreate(&stream)); unsigned int* errors;OK(cudaMalloc(&errors,4));OK(cudaMemsetAsync(errors,0,4,stream));
 for(int C: {7056,3529}) {
  long long count=(long long)4096*128*C,mb=(long long)64*128*C;
  float *src,*ref,*scratch;OK(cudaMalloc(&src,count*4));OK(cudaMalloc(&ref,count*4));OK(cudaMalloc(&scratch,mb*4));
  fill<<<(count+255)/256,256,0,stream>>>(src,count,C==3529);
  char fixture_name[4096];snprintf(fixture_name,sizeof(fixture_name),"%s/%d-fixture.bin",argv[1],C);
  FILE* fixture=fopen(fixture_name,"rb");if(!fixture)return 6;
  float* fixture_host=(float*)malloc(8*C*4);if(fread(fixture_host,4,8*C,fixture)!=8*C)return 7;fclose(fixture);
  for(int sample=0;sample<8;sample++)OK(cudaMemcpyAsync(src+(long long)(sample*8*64)*C,fixture_host+sample*C,C*4,cudaMemcpyHostToDevice,stream));
  OK(cudaStreamSynchronize(stream));free(fixture_host);
  baseline<<<(count+255)/256,256,0,stream>>>(ref,src,C);
  char name[4096];snprintf(name,sizeof(name),"%s/%d-reference.bin",argv[1],C);FILE* fref=fopen(name,"wb");
  snprintf(name,sizeof(name),"%s/%d-gather.bin",argv[1],C);FILE* fg=fopen(name,"wb");if(!fref||!fg)return 4;
  float* host=(float*)malloc(C*4);
  // Two complete rotations; second reuse reverses block order. One stream owns scratch.
  for(int pass=0;pass<2;pass++)for(int e=0;e<2;e++)for(int m=0;m<32;m++){
   int block=pass ? 63-(e*32+m) : e*32+m,off=block*64;
   metta_gather_rollout<<<(mb+255)/256,256,0,stream>>>(scratch,src,128,4096,C,off,mb);
   equal_bits<<<(mb+255)/256,256,0,stream>>>(scratch,ref+(long long)off*128*C,mb,errors);
   // Sample eight distinct blocks after bit comparison; copied before scratch reuse.
   if(pass==0 && block%8==0){OK(cudaMemcpyAsync(host,ref+(long long)off*128*C,C*4,cudaMemcpyDeviceToHost,stream));OK(cudaStreamSynchronize(stream));fwrite(host,4,C,fref);
    OK(cudaMemcpyAsync(host,scratch,C*4,cudaMemcpyDeviceToHost,stream));OK(cudaStreamSynchronize(stream));fwrite(host,4,C,fg);}
  }
  OK(cudaStreamSynchronize(stream));free(host);fclose(fref);fclose(fg);OK(cudaFree(src));OK(cudaFree(ref));OK(cudaFree(scratch));
 }
 unsigned int n=0;OK(cudaMemcpy(&n,errors,4,cudaMemcpyDeviceToHost));OK(cudaFree(errors));OK(cudaStreamDestroy(stream));
 if(n){fprintf(stderr,"bit mismatches %u\n",n);return 5;}return 0;
}
'''


def model_parity(directory, build):
    import os
    import numpy as np
    # Use the same acting transform configuration as the qualified native trainer.
    from integrations.policy_execution import runtime_environment
    inputs = Path('/work/input')
    metadata = json.loads((inputs/'assets/cold/asset.json').read_text())
    source = Path(__file__).resolve().parents[1]
    os.environ.update(runtime_environment(source, directory, metadata['sampler']))
    import jax
    import jax.numpy as jnp
    from integrations.direct_spatial_optimization import install
    import metta_training.native_fabric as native
    if not getattr(native.NativeFabricPolicy, '_generals_direct_spatial', False):
        install(native)
    devices = jax.devices('gpu')
    if len(devices) != 1:
        raise ValueError('Exactly one GPU required')
    with jax.default_device(jax.devices('cpu')[0]):
        policy = native.NativeFabricPolicy(json.dumps(build['config']['fabric']))
    from integrations.native_spatial_asset import abi_digest
    if abi_digest(policy) != metadata['abi_sha256'] or policy.buffers.parameter_words != metadata['parameter_count']:
        raise ValueError('Gather admission policy ABI differs')
    if build['config']['fabric'] != metadata['fabric'] or digest(source/'integrations/generals_fabric.py') != metadata['factory_source_sha256']:
        raise ValueError('Model or factory differs from selected source')
    parameters = jax.device_put(np.fromfile(inputs/'assets/cold/policy.bin', '<f4'), devices[0])
    if digest(inputs/'assets/cold/policy.bin') != metadata['policy_sha256']:
        raise ValueError('Source weights differ')
    def load(which):
        obs = np.fromfile(directory/f'7056-{which}.bin', '<f4').reshape(8,1,7056)
        mask = np.fromfile(directory/f'3529-{which}.bin', '<f4').reshape(8,1,3529)
        return jax.device_put(obs, devices[0]), jax.device_put(mask, devices[0])
    observations, masks = load('reference')
    gathered, gathered_masks = load('gather')
    return dict(**ppo_parity(policy, parameters, observations, masks, gathered, gathered_masks),
                policy_sha256=metadata['policy_sha256'], abi_sha256=metadata['abi_sha256'])


def ppo_parity(policy, parameters, observations, masks, gathered, gathered_masks):
    """Exercise the native forward tape and production PPO backward API."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    terminals = jnp.asarray(np.arange(8).reshape(8,1)%2, dtype=jnp.float32)
    with jax.default_matmul_precision('highest'):
        initial, tape_a = policy._forward_arrays(parameters, observations, terminals, 8, 1, False)
        candidate, tape_b = policy._forward_arrays(parameters, gathered, terminals, 8, 1, False)
    logits = jnp.where(masks > 0, initial[...,:3529], -1e9)
    actions = jnp.argmax(logits, axis=-1)
    old_logp = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[...,None], -1)[...,0]
    old_logp = jax.lax.stop_gradient(old_logp - jnp.log(jnp.asarray([.5,.9,1.1,1.5,.5,.9,1.1,1.5]).reshape(8,1)))
    advantage = jnp.linspace(-1,1,8).reshape(8,1)
    returns = jax.lax.stop_gradient(initial[...,3529] + advantage*.1)
    def objective(out, mask):
        logp = jax.nn.log_softmax(jnp.where(mask>0,out[...,:3529],-1e9))
        ratio = jnp.exp(jnp.take_along_axis(logp,actions[...,None],-1)[...,0]-old_logp)
        loss = -jnp.mean(jnp.minimum(ratio*advantage,jnp.clip(ratio,.8,1.2)*advantage)) + .5*jnp.mean((out[...,3529]-returns)**2)
        return loss, jnp.exp(logp)
    # The native boundary performs concrete finite checks. Differentiate the PPO
    # output loss, then feed its cotangents to the same native backward as training.
    operation = jax.jit(jax.value_and_grad(objective, has_aux=True))
    with jax.default_matmul_precision('highest'):
        (loss_a, probabilities_a), cotangents_a = operation(initial, masks)
        (loss_b, probabilities_b), cotangents_b = operation(candidate, gathered_masks)
        gradient_a = policy.backward_device_arrays(tape_a, cotangents_a[...,:3529], cotangents_a[...,3529])
        gradient_b = policy.backward_device_arrays(tape_b, cotangents_b[...,:3529], cotangents_b[...,3529])
    for label,a,b in [('loss',loss_a,loss_b),('probabilities',probabilities_a,probabilities_b),('parameter_gradient',gradient_a,gradient_b)]:
        a,b=np.asarray(a),np.asarray(b)
        if not np.isfinite(a).all() or not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise ValueError('Gather changed spatial PPO '+label)
    if not np.any(np.asarray(gradient_a)):
        raise ValueError('Degenerate zero-gradient admission')
    return dict(states=8, parameter_words=int(parameters.size), mixed_terminals=True,
                probabilities_bitwise=True, ppo_loss_bitwise=True, parameter_gradient_bitwise=True,
                gradient_api='NativeFabricPolicy.backward_device_arrays')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    from integrations.puffer_rollout_memory import validate
    validate(args.build)
    build=json.loads((args.build/'build.json').read_text())
    if digest(args.build/'puffer') != build['binary_sha256'] or build['model_state_words'] != 0:
        raise ValueError('Actual build identity differs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    header=args.build/'source/src/metta_rollout_memory.cuh'
    with tempfile.TemporaryDirectory(prefix='gather-gpu-') as name:
        directory=Path(name)
        import jax
        import numpy as np
        from integrations.audit_spatial_checkpoint_serving_parity import verified_views
        with jax.default_device(jax.devices('cpu')[0]):
            observations,masks,labels=verified_views(Path('/work/input/leader-root'),[0,6,7,10],{0,25,99,100,150,200})
        if len(observations)<8:
            raise ValueError('Require eight authenticated public replay observations')
        observations[:8].astype('<f4').tofile(directory/'7056-fixture.bin')
        masks[:8].astype('<f4').tofile(directory/'3529-fixture.bin')
        fixtures={str(c):digest(directory/f'{c}-fixture.bin') for c in (7056,3529)}
        (directory/'admission.cu').write_text('#include <initializer_list>\n'+CUDA)
        subprocess.run(['nvcc','-std=c++17','-O2','-I',str(header.parent),str(directory/'admission.cu'),'-o',str(directory/'admission')],check=True,timeout=120)
        subprocess.run([str(directory/'admission'),str(directory)],check=True,timeout=180)
        model=model_parity(directory,build)
    report=dict(schema='generals-minibatch-gather-gpu-admission-v1',passed=True,backend='gpu',
                build_sha256=digest(args.build/'build.json'),binary_sha256=build['binary_sha256'],
                header_sha256=digest(header),rollout_memory_receipt_sha256=digest(args.build/'rollout-memory.json'),
                generated_pufferl_sha256=digest(args.build/'source/src/pufferl.cu'),audit_module_sha256=digest(__file__),
                all_64_blocks_bitwise=True,scratch_reuse_passes=2,geometry=dict(agents=4096,horizon=128,minibatch_rows=64,features=[7056,3529]),
                model=model,fixture_sha256=fixtures,fixture_labels=labels[:8],scope='Actual compiled gather header, exact full-size float32 observations/binary masks; spatial PPO parity on eight authenticated public replay states. No claim of end-to-end training equivalence.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()

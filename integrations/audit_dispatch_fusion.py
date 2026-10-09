"""Separate-process production dispatch admission with retained gradient controls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural(source):
    tree=ast.parse((source/'integrations/direct_spatial_optimization.py').read_text())
    names=('DirectSpatial','direct_backward')
    return {name:hashlib.sha256(ast.dump(next(n for n in ast.walk(tree)
            if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name),include_attributes=False).encode()).hexdigest() for name in names}


def run(args):
    import integrations
    args.source=args.source.resolve()
    integrations.__path__.insert(0,str(args.source/'integrations'))
    import os
    from integrations.policy_execution import runtime_environment
    asset=args.asset or args.input/'assets/cold/asset.json'
    metadata=json.loads(asset.read_text())
    os.environ.update(runtime_environment(args.source,args.output,metadata['sampler']))
    import jax
    import jax.numpy as jnp
    import numpy as np
    from integrations.audit_spatial_checkpoint_serving_parity import verified_views
    from integrations.direct_spatial_optimization import install, DirectTape
    import integrations.direct_spatial_optimization as implementation
    import metta_training.native_fabric as native
    from integrations.native_spatial_asset import abi_digest
    assert Path(implementation.__file__).resolve()==args.source/'integrations/direct_spatial_optimization.py'
    devices=jax.devices(args.backend)
    assert len(devices)==1 and not args.output.exists()
    args.output.mkdir(parents=True)
    if not getattr(native.NativeFabricPolicy,'_generals_direct_spatial',False):install(native)
    with jax.default_device(jax.devices('cpu')[0]):
        policy=native.NativeFabricPolicy(json.dumps(metadata['fabric']))
        observations,masks,labels=verified_views(args.input/'leader-root',[0,6,7,10],{0,25,99,100,150,200})
    assert abi_digest(policy)==metadata['abi_sha256'] and policy.teacher is None and policy.teacher_phase.ppo_coefficient==1
    parameters=jax.device_put(np.fromfile(asset.parent/'policy.bin','<f4'),devices[0])
    assert sha(asset.parent/'policy.bin')==metadata['policy_sha256']
    obs=jax.device_put(observations[:8].reshape(8,1,7056),devices[0]); mask=jax.device_put(masks[:8].reshape(8,1,3529),devices[0])
    terminals=jax.device_put(np.arange(8,dtype=np.float32).reshape(8,1)%2,devices[0])
    def save(name,value):
        value=np.asarray(value); assert np.isfinite(value).all()
        np.save(args.output/(name+'.npy'),value,allow_pickle=False)
        return value
    with jax.default_matmul_precision('highest'):
        acting,tape=policy._forward_arrays(parameters,obs,terminals,8,1,False)
    for name,value in [('parameters',tape.parameters),('observations',tape.observations),('masks',mask),('raw',tape.predictions),('acting',acting)]:save(name,value)
    actions=jnp.argmax(jnp.where(mask,acting[...,:3529],-1e9),axis=-1)
    old=jnp.take_along_axis(jax.nn.log_softmax(jnp.where(mask,acting[...,:3529],-1e9)),actions[...,None],-1)[...,0]
    old=old-jnp.log(jnp.asarray([.5,.9,1.1,1.5,.5,.9,1.1,1.5]).reshape(8,1))
    advantages=jnp.linspace(-1,1,8).reshape(8,1);returns=acting[...,3529]+advantages*.1
    controls={'actions':actions,'old_logp':old,'returns':returns}
    for name,value in controls.items():
        if args.baseline:controls[name]=jax.device_put(np.load(args.baseline/(name+'.npy'),allow_pickle=False),devices[0])
        save(name,controls[name])
    def objective(out):
        logp=jax.nn.log_softmax(jnp.where(mask,out[...,:3529],-1e9))
        ratio=jnp.exp(jnp.take_along_axis(logp,controls['actions'][...,None],-1)[...,0]-controls['old_logp'])
        return -jnp.mean(jnp.minimum(ratio*advantages,jnp.clip(ratio,.8,1.2)*advantages))+.5*jnp.mean((out[...,3529]-controls['returns'])**2),jnp.exp(logp)
    (loss,probabilities),cot=jax.jit(jax.value_and_grad(objective,has_aux=True))(acting)
    save('loss',loss);save('probabilities',probabilities);save('cotangents',cot)
    common_tape,common_cot=tape,cot
    if args.baseline:
        previous=json.loads((args.baseline/'admission.json').read_text())
        for name,digest in previous['files'].items():assert sha(args.baseline/name)==digest
    if args.baseline:
        common_tape=DirectTape(parameters,obs,jax.device_put(np.load(args.baseline/'raw.npy',allow_pickle=False),devices[0]))
        common_cot=jax.device_put(np.load(args.baseline/'cotangents.npy',allow_pickle=False),devices[0])
    save('common-raw',common_tape.predictions);save('common-cotangents',common_cot)
    for name,t,c in [('own',tape,cot),('common',common_tape,common_cot)]:
        for index in range(3):
            with jax.default_matmul_precision('highest'):
                gradient=policy.backward_device_arrays(t,c[...,:3529],c[...,3529])
            value=save(f'{name}-gradient-{index}',gradient)
            assert value.shape==(578860,) and np.any(value)
    receipt=dict(audit_module_sha256=sha(Path(__file__)),sampler=metadata['sampler'],teacher_ppo_coefficient=policy.teacher_phase.ppo_coefficient,backend=args.backend,policy_sha256=metadata['policy_sha256'],abi_sha256=metadata['abi_sha256'],
                 source_sha256=sha(Path(implementation.__file__)),structure=structural(args.source),
                 sampling_source_sha256=sha(args.source/'integrations/spatial_action_sampling.py'),
                 fixture_labels=labels[:8],files={p.name:sha(p) for p in args.output.glob('*.npy')})
    (args.output/'admission.json').write_text(json.dumps(receipt,indent=2)+'\n')


def compare(baseline,candidate,output):
    import numpy as np
    receipts=[json.loads((p/'admission.json').read_text()) for p in (baseline,candidate)]
    for key in ('sampler','teacher_ppo_coefficient','backend','policy_sha256','structure','sampling_source_sha256','fixture_labels'):
        assert receipts[0][key]==receipts[1][key],key
    for path,receipt in zip((baseline,candidate),receipts):
        for name,digest in receipt['files'].items():assert sha(path/name)==digest
    def load(path,name):return np.load(path/(name+'.npy'),allow_pickle=False)
    def delta(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
        d=a.astype(np.float64)-b.astype(np.float64)
        return dict(max_abs=float(np.max(np.abs(d))),l2=float(np.linalg.norm(d)),reference_l2=float(np.linalg.norm(a.astype(np.float64))))
    comparisons={n:delta(load(baseline,n),load(candidate,n)) for n in ('observations','masks','raw','acting','probabilities','loss','cotangents')}
    for name in ('observations','masks'):
        assert load(baseline,name).tobytes()==load(candidate,name).tobytes()
    assert comparisons['probabilities']['max_abs']<=1e-5
    for path in (baseline,candidate):
        probabilities=load(path,'probabilities');mask=load(path,'masks')
        assert np.max(np.abs(probabilities.sum(-1)-1))<=1e-5 and not np.any(probabilities[~mask.astype(bool)])
    assert np.array_equal(load(baseline,'probabilities').argmax(-1),load(candidate,'probabilities').argmax(-1))
    for name in ('parameters','observations','masks','common-raw','common-cotangents','actions','old_logp','returns'):
        a,b=load(baseline,name),load(candidate,name)
        assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
    gradients={}
    for mode in ('own','common'):
        arrays=[(f'{arm}-{i}',load(p,f'{mode}-gradient-{i}')) for arm,p in [('baseline',baseline),('candidate',candidate)] for i in range(3)]
        assert all(a.shape==(578860,) and np.isfinite(a).all() and np.any(a) for _,a in arrays)
        gradients[mode]={a+' vs '+b:delta(x,y) for i,(a,x) in enumerate(arrays) for b,y in arrays[i+1:]}
    output.write_text(json.dumps(dict(passed=True,backend=receipts[0]['backend'],comparisons=comparisons,gradients=gradients,
        admissions=[sha(p/'admission.json') for p in (baseline,candidate)],
        scope='Existing probability tolerance and identical chosen actions; unchanged backward AST and sampling code with common raw predictions/cotangents. Gradient variability quantified, not required bitwise.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--asset',type=Path);p.add_argument('--input',type=Path,default=Path('/work/input'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--compare',type=Path,nargs=2)
    p.add_argument('--backend',choices=('gpu','cpu'),default='gpu');a=p.parse_args()
    if a.compare:compare(*a.compare,a.output)
    else:run(a)


if __name__=='__main__':main()

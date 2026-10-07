"""Bounded same-allocation ABBA execution diagnosis; never qualifies a policy."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

PLAN = json.loads(Path(__file__).with_name('bridge_profile_plan.json').read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')
def diagnostic(path):
    spec=importlib.util.spec_from_file_location('bridge_execution_diagnostic',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def build(args):
    from integrations.launch_spatial_selfplay_training import load_pinned_trainer
    trainer=load_pinned_trainer()
    expected=args.inputs/'sources'/args.variant/'integrations/puffer_coworld_frozen_transfer.py'
    if Path(trainer.__file__).resolve()!=expected.resolve():
        raise ValueError('Build loaded another source variant')
    write(args.output/'trainer-binding.json',dict(variant=args.variant,trainer_source_sha256=sha(expected),
          source_revision=read(args.inputs/'source-revisions.json')[args.variant]))
    helper=diagnostic(args.inputs/'diagnostic/bridge_execution_diagnostic.py')
    original=trainer.install_driver
    def install(source, **kwargs):
        environment=original(source, **kwargs)
        receipt=helper.instrument_build(source)
        write(args.output/'instrumentation.json',receipt)
        return environment
    trainer.install_driver=install
    trainer.build_puffer(args.output/'build',trainer.BuildConfig.model_validate(read(args.output/'build-config.json')))
    generated=args.output/'build/source/src'
    write(args.output/'generated-source.json',{name:sha(generated/name) for name in
          ('pufferl.cu','metta_device_environment.cuh','metta_fabric.cuh','bridge_execution_profile.cuh','bridge_lifetime_probe.cu')})
    lifetime=helper.compile_and_run_lifetime_probe(args.output/'build/source',args.output/'lifetime',
                                         4 if args.variant=='baseline' else 1)
    assert lifetime['passed'] and lifetime['iterations']==64 and lifetime['arrays_per_iteration']==4
    assert lifetime['output_fences_per_iteration']==(4 if args.variant=='baseline' else 1)
    assert lifetime['bitwise_equal'] and lifetime['owners_released_after_fence'] and lifetime['cross_stream_producer']
    write(args.output/'lifetime-audit.json',lifetime)


def run_build(inputs, output, variant, source, sampler):
    from integrations.policy_execution import runtime_environment
    env=runtime_environment(source,output,sampler)
    with (output/'build-process.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u',__file__,'build','--inputs',str(inputs),
             '--output',str(output),'--variant',variant],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=600)
            if code:raise RuntimeError(f'{variant} build/lifetime probe failed, exit {code}')
        finally:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()


def audit_profile(rows):
    import math
    expected = {'environment_input_fence':128, 'environment_output':128,
                'pufferl_forward_step':128, 'learner_rollout_native_forward':128,
                'optimization_epoch':1, 'learner_train_forward':32}
    totals = {}
    for row in rows:
        epoch, label = row['epoch'], row['label']
        assert epoch in (-1,0,1,2,3) and row['pending_events']==0
        assert label in expected and (epoch!=-1 or label=='environment_output')
        assert isinstance(row['calls'],int) and row['calls']>0
        assert all(math.isfinite(row[k]) and row[k]>=0 for k in ('host_seconds','cuda_stream_span_ms'))
        totals[epoch,label]=totals.get((epoch,label),0)+row['calls']
    for epoch in range(4):
        for label,count in expected.items():assert totals.get((epoch,label))==count,(epoch,label,totals)
    return totals


def run_audit(out, config, build_config):
    from integrations.monitor_coworld_steady_interval import completed_epoch_times
    import re
    expected=PLAN['steps_per_run']
    assert read(out/'run/completed.json')['trained_timesteps']==expected
    from integrations.learner_checkpoint import LearnerCheckpoint
    checkpoint=out/f'run/checkpoints/metta_generals/run/{expected:016d}.bin'
    parameters=read(Path(config['initialize']['asset']))['parameter_count']
    learner=LearnerCheckpoint.read(Path(str(checkpoint)+'.learner'),parameters)
    assert learner.agent_steps==expected and learner.epoch==4
    assert checkpoint.is_file()
    assert sha(out/'run/initial-policy.bin')==PLAN['source_policy_sha256']
    assert not (out/'run/initial-policy.bin.learner').exists()
    text=(out/'run/console.log').read_text()
    assert f'DEVICE_ACTION_MASK_AUDIT actions={expected} illegal=0' in text
    rewards=[json.loads(x.split(' ',1)[1]) for x in text.splitlines() if x.startswith('DEVICE_REWARD_AUDIT ')]
    assert rewards and rewards[-1]['agent_steps']==expected
    assert all(not rewards[-1][k] for k in ('nonfinite_rewards','native_clipped_rewards','native_clipped_terminal_rewards'))
    population=read(out/f"run/environments/{config['seed']}/spatial-opponent-population.json")
    options=build_config['python_environment']['options']
    frozen=[sha(Path(p)/'policy.bin') for p in options['frozen_bundles']]
    assert population['frozen_policy_sha256']==frozen and population['opponent_weights']==options['opponent_weights']
    names={'frozen_'+h[:12] for h in frozen}|set(options['scripted_opponents'])
    assert set(population['counts'])==names and all(x['0']==x['1']>0 for x in population['counts'].values())
    rotation=[tuple(map(int,x)) for x in re.findall(r'MINIBATCH_ROTATION epoch=(\d+) start_block=(\d+) total_minibatches=(\d+) total_blocks=(\d+) rows_per_block=(\d+) rule=epoch_times_updates_mod_blocks',text)]
    assert rotation==[(e,e*32%64,32,64,64) for e in range(4)]
    epochs=completed_epoch_times(text);assert all(e in epochs for e in (1,2,3,4))
    measured_seconds=epochs[4]-epochs[2];assert measured_seconds>0
    profile=[json.loads(x.split(' ',1)[1]) for x in text.splitlines() if x.startswith('EXECUTION_PROFILE ')]
    audit_profile(profile)
    report=dict(diagnostic_only=True,qualified=False,steps=expected,warmup_epochs=2,
                epoch_uptime=epochs,measured_steps=1048576,measured_seconds=measured_seconds,
                measured_sps=1048576/measured_seconds,rotation=rotation,reward_audit=rewards[-1],
                opponent_counts_by_seat=population['counts'],execution_profile=profile)
    write(out/'diagnostic-audit.json',report)
    return report


def prepare(inputs, out):
    from integrations.classic_contract import validate_training_contract
    from integrations.native_spatial_asset import load_asset
    from integrations.launch_spatial_selfplay_training import source_sampling_gate_report,validate_sampling_gate
    from integrations.policy_execution import execute,runtime_environment
    inputs=inputs.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',PLAN)
    for name,digest in read(inputs/'manifest.json').items():
        if sha(inputs/name)!=digest:raise ValueError('Input identity differs: '+name)
    asset_path=inputs/'assets/cold/asset.json';asset=load_asset(asset_path,manifest_sha256=sha(asset_path))
    assert asset.metadata['policy_sha256']==PLAN['source_policy_sha256'];sampler=asset.metadata['sampler']
    gate=source_sampling_gate_report(inputs/'source-sampling')
    effective=dict(full_action_temperature=1.0,route_half_weight=0.0,**sampler)
    assert gate['source_sha256']==gate['opponent_sha256']==PLAN['source_policy_sha256']
    assert gate['sampler']==gate['opponent_sampler']==effective
    assert gate['match_seed']==11008101 and gate['sample_seed']==11008103
    build_config=read(inputs/'build-config.json');config=read(inputs/'config.json')
    build_config['python_environment']['options']['classic_siege_workers']=8
    config.update(seed=PLAN['training_seed'],total_timesteps=PLAN['steps_per_run'],
                  initialize=dict(asset=str(asset_path),manifest_sha256=sha(asset_path),restore_learner=False))
    assert config['overrides']['vec.total_agents']==4096 and config['overrides']['train.horizon']==128
    assert config['overrides']['train.minibatch_size']==8192 and config['overrides']['train.replay_ratio']==.5
    write(out/'classic-contract.json',validate_training_contract(build_config,config))
    write(out/'input-binding.json',dict(manifest_sha256=sha(inputs/'manifest.json'),
          source_policy_sha256=PLAN['source_policy_sha256'],source_asset_sha256=sha(asset_path),
          source_revisions=read(inputs/'source-revisions.json'),diagnostic_sha256=sha(inputs/'diagnostic/bridge_execution_diagnostic.py')))
    return build_config, config, sampler, gate


def run(args):
    from integrations.policy_execution import execute,runtime_environment
    from integrations.launch_spatial_selfplay_training import validate_sampling_gate
    from integrations.slurm_s3_job import visible_gpu_identity,verify_gpu_idle
    inputs=args.inputs.resolve();out=args.output.resolve()
    build_config,config,sampler,gate=prepare(inputs,out)
    write(out/'gpu-preflight.json',verify_gpu_idle(visible_gpu_identity()))
    for variant in ('baseline','candidate'):
        target=out/variant;target.mkdir();write(target/'build-config.json',build_config)
        print('PROBE_PHASE '+json.dumps(dict(phase='build',variant=variant,event='start')),flush=True)
        run_build(inputs,target,variant,inputs/'sources'/variant,sampler)
        print('PROBE_PHASE '+json.dumps(dict(phase='build',variant=variant,event='complete')),flush=True)
    reports=[]
    for index,variant in enumerate(PLAN['sequence']):
        target=out/f'{index}-{variant}';target.mkdir();source=inputs/'sources'/variant
        write(target/'config.json',config);write(target/'sampling-gate.json',gate)
        argv=['train','--build',out/variant/'build','--config',target/'config.json','--output',target/'run']
        validate_sampling_gate(['launcher',*map(str,argv)],runtime_environment(source,target,sampler))
        print('PROBE_PHASE '+json.dumps(dict(phase='train',index=index,variant=variant,event='start')),flush=True)
        execute('launch_spatial_selfplay_training',argv,source=source,output=target,sampler=sampler,
                name='train',seconds=480,training_config=target/'config.json',startup_seconds=420,diagnostic_profile=True)
        reports.append(dict(index=index,variant=variant,**run_audit(target,config,build_config)))
        print('PROBE_PHASE '+json.dumps(dict(phase='train',index=index,variant=variant,event='complete',sps=reports[-1]['measured_sps'])),flush=True)
    write(out/'report.json',dict(diagnostic_only=True,qualified=False,sequence=PLAN['sequence'],runs=reports,
         conclusion='Instrumented execution evidence only; full uninstrumented >=30000 SPS qualification remains required.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('build','run'));p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=('baseline','candidate'));args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError(f'Probe signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    from integrations.cuda_runtime_binding import configure
    configure()
    globals()[args.phase](args)
if __name__=='__main__':main()

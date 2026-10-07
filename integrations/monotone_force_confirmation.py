"""Conditional independent frozen confirmation; no upload, submission or training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from integrations.monotone_force_trial import selected
from integrations.analyze_spatial_population_pair import compare

PLAN=json.loads(Path(__file__).with_name('monotone_force_confirmation_plan.json').read_text())

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def terminal_admission(audit):
    if (audit.get('schema') != PLAN['required_terminal_audit_schema']
            or audit.get('job_id') != PLAN['required_job_id']
            or audit.get('provider_status') != 'succeeded'
            or any(audit.get(k) is not True for k in ('technical_success','qualification_verified','collection_complete','selected'))):
        raise ValueError('Development terminal has not qualified independent confirmation')

def verify_development(audit_path, expected_sha, source_input):
    if sha(audit_path)!=expected_sha:raise ValueError('Independent terminal audit hash differs')
    audit=read(audit_path);terminal_admission(audit)
    paths={k:Path(v) for k,v in audit['paths'].items()}
    for role,field in [('terminal','terminal_sha256'),('provider_archive','artifact_sha256'),
                       ('collection','collection_sha256'),('results_archive','results_archive_sha256'),('completed','completed_sha256')]:
        if sha(paths[role])!=audit[field]:raise ValueError('Verified terminal evidence changed: '+role)
    terminal=read(paths['terminal'])
    if terminal['job_id'] != PLAN['required_job_id'] or terminal['status']!='succeeded':
        raise ValueError('Actual provider terminal differs from audit')
    collection=read(paths['collection']);completed=read(paths['completed']);root=paths['generals']
    if (collection.get('complete') is not True or completed.get('selected') is not True
            or completed['results_sha256']!=audit['results_archive_sha256']
            or completed['selection_sha256']!=sha(root/'selection.json')
            or completed['plan_sha256']!=sha(root/'plan.json')):
        raise ValueError('Completed result/collection binding differs')
    for name,digest in collection['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root/relative)!=digest:
            raise ValueError('Collected experiment file changed: '+name)
    from integrations.monotone_force_trial import PLAN as development_plan
    if read(root/'plan.json') != development_plan:raise ValueError('Development preregistration differs')
    reports=[compare(root/('heldout-'+before),root/'heldout-candidate',seed=12001111,resamples=10000)
             for before in ('source','control')]
    if not selected(reports):raise ValueError('Recomputed development comparison fails selection')
    if read(root/'selection.json')['selected'] is not True:raise ValueError('Development selection was negative')
    source_input=source_input.resolve()
    bundles={'source':source_input/'bundles/cold',**{arm:root/arm/'continuation/bundle' for arm in ('control','candidate')}}
    from integrations.spatial_policy_bundle import SpatialPlayerPolicy
    policies={name:SpatialPlayerPolicy(path) for name,path in bundles.items()}
    source=policies['source'].asset.metadata
    if source['policy_sha256'] != PLAN['source_policy_sha256']:raise ValueError('Source checkpoint differs')
    identities={}
    for name,policy in policies.items():
        metadata=policy.asset.metadata
        if (metadata['sampler']!=source['sampler'] or metadata['fabric']!=source['fabric']
                or metadata['model_sha256']!=source['model_sha256'] or metadata['abi_sha256']!=source['abi_sha256']):
            raise ValueError('Frozen model/sampler identity differs: '+name)
        if name!='source':
            parity=read(root/name/'continuation/serving-parity.json')
            trained=read(root/name/'continuation/asset/asset.json')
            if (trained['policy_sha256']!=metadata['policy_sha256']
                    or parity['checkpoint_sha256']!=metadata['policy_sha256']
                    or parity['public_states']<=0 or parity['matching_top_actions']!=parity['public_states']):
                raise ValueError('Trained native/serving evidence differs: '+name)
        identities[name]=dict(path=str(bundles[name]),policy_sha256=metadata['policy_sha256'],
             files={p.name:sha(p) for p in sorted(bundles[name].iterdir()) if p.is_file()})
    build=root/'control/build/build.json';options=read(build)['config']['python_environment']['options']
    pool=[]
    for index in range(10):
        bundle=source_input/'bundles/frozen'/str(index)
        digest=sha(bundle/'policy.bin')
        expected=read(root/'heldout-source/evaluation.json')['frozen_policy_sha256'][index]
        if digest!=expected:raise ValueError('Original opponent pool differs')
        pool.append(dict(path=str(bundle),files={p.name:sha(p) for p in sorted(bundle.iterdir()) if p.is_file()}))
    return dict(schema=PLAN['schema'],development_job=PLAN['required_job_id'],development_audit_sha256=expected_sha,
                development_results_sha256=audit['results_archive_sha256'],plan=PLAN,bundles=identities,pool=pool,
                sampler=source['sampler'],population_build=str(build),population_build_sha256=sha(build),
                opponent_weights=options['opponent_weights'],scripted_opponents=options['scripted_opponents'],
                source=str(Path(__file__).resolve().parents[1]),development_comparisons=reports)

def prepare(audit,expected_sha,source_input,output):
    if output.exists():raise ValueError('Confirmation configuration must be a new file')
    config=verify_development(audit,expected_sha,source_input)
    # No directory or runnable config is created until every condition passes.
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(config,indent=2)+'\n')
    return config

def run(config_path,output):
    config=read(config_path)
    if config['schema']!=PLAN['schema'] or config['plan']!=PLAN or config['development_job']!=PLAN['required_job_id']:
        raise ValueError('Confirmation configuration differs from preregistration')
    for bundle in list(config['bundles'].values())+config['pool']:
        for name,digest in bundle['files'].items():
            if sha(Path(bundle['path'])/name)!=digest:raise ValueError('Frozen confirmation input changed')
    if sha(config['population_build'])!=config['population_build_sha256']:raise ValueError('Population build changed')
    if output.exists():raise ValueError('Confirmation output must be new')
    from integrations.cuda_runtime_binding import configure
    configure()
    from integrations.policy_execution import execute
    output.mkdir(parents=True)
    for arm in PLAN['arms']:
        execute('evaluate_spatial_population',['--bundle',config['bundles'][arm]['path'],
            '--population-build',config['population_build'],'--games',4096,'--pool-size',4096,
            '--seed',PLAN['evaluation_seed'],'--sample-seed',PLAN['evaluation_sample_seed'],
            '--destination-audit','--output',output/arm],source=Path(config['source']),output=output,
            sampler=config['sampler'],name='confirm-'+arm,seconds=900)
    reports=[compare(output/arm,output/'candidate',seed=PLAN['bootstrap_seed'],resamples=10000) for arm in ('source','control')]
    result=dict(schema=PLAN['schema'],confirmed=selected(reports),configuration_sha256=sha(config_path),
                comparisons=reports,requires_hosted_confirmation=True)
    (output/'confirmation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    prepare_args=sub.add_parser('prepare');prepare_args.add_argument('--audit',type=Path,required=True)
    prepare_args.add_argument('--audit-sha256',required=True);prepare_args.add_argument('--source-input',type=Path,required=True)
    prepare_args.add_argument('--output',type=Path,required=True)
    run_args=sub.add_parser('run');run_args.add_argument('--config',type=Path,required=True);run_args.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.audit,a.audit_sha256,a.source_input,a.output)
    else:run(a.config,a.output)

if __name__=='__main__':main()

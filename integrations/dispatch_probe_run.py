"""One-allocation ABBA diagnostic; never authorizes long training or selects a policy."""
import argparse
import copy
import json
import re
from pathlib import Path

from integrations.policy_execution import execute
from integrations.policy_trial import digest, read, write, rotation_audit
from integrations.training_inputs import CONFIG, ASSETS

PLAN = Path(__file__).with_name('dispatch_probe_plan.json')
ORDER = ('baseline', 'fused', 'fused', 'baseline')


def prepare(inputs, output):
    plan = read(PLAN)
    if (plan['status'] != 'sealed before launch'
            or plan['order'] != list(ORDER)
            or plan['per_run_steps'] != 2097152 or plan['per_run_seconds'] != 480
            or plan['warmup_epochs'] != 2
            or plan['runtime_bound'] != dict(provider_minutes=60, execution_minutes=58,
                aggregate_minutes_from_first_allocation=60, max_restarts=0)
            or plan['baseline_commit'] != '589bfab160c583798f69892e4dd9387dae7f7f85'
            or not re.fullmatch('[a-f0-9]{40}', plan['fused_commit'] or '')
            or not re.fullmatch('[a-f0-9]{64}', plan['input_manifest_sha256'] or '')
            or digest(inputs/'input-manifest.json') != plan['input_manifest_sha256']):
        raise ValueError('Unsealed or changed ABBA plan')
    manifest = read(inputs/'input-manifest.json')
    required = {'baseline-source.json', 'fused-source.json', 'config.json', 'build-config.json'}
    required |= {'abi-migration/'+name for name in ('baseline.json', 'candidate.json', 'proof.json', 'migration.json', 'migrate.py')}
    required |= {'fused/'+str(p.relative_to(inputs/'fused')) for p in (inputs/'fused').rglob('*') if p.is_file()}
    if not required <= manifest.keys():
        raise ValueError('Missing fixed input bindings')
    for name, expected in manifest.items():
        path = (inputs/name).resolve()
        if (inputs/name).is_symlink() or not path.is_relative_to(inputs.resolve()) or digest(path) != expected:
            raise ValueError('Input changed: ' + name)
    # Require complete source closures, not a manifest containing selected files only.
    for arm in ('baseline', 'fused'):
        source = inputs/'sources'/arm
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('Source closure contains a symlink')
        files = {str(p.relative_to(inputs)) for p in source.rglob('*') if p.is_file()}
        if not files or not files <= manifest.keys():
            raise ValueError('Incomplete source closure: ' + arm)
        if read(inputs/(arm+'-source.json'))['commit'] != plan[arm+'_commit']:
            raise ValueError('Source commit binding differs')
    if digest(inputs/'sources/baseline/integrations/generals_fabric.py') != digest(inputs/'sources/fused/integrations/generals_fabric.py'):
        raise ValueError('ABBA source factories differ')
    build = copy.deepcopy(read(CONFIG/'build-config.json'))
    build['python_environment']['options']['coworld_position_probability'] = 0.0
    config = read(CONFIG/'training-config.json')
    config['total_timesteps'] = 2097152
    if read(inputs/'build-config.json') != build or read(inputs/'config.json') != config:
        raise ValueError('Diagnostic configuration differs')
    from integrations.native_spatial_asset import load_asset
    from integrations.classic_contract import validate_training_contract
    for item in ASSETS:
        asset = load_asset(inputs/Path(item['destination']).relative_to('/work/input')/'asset.json',
                           manifest_sha256=item['manifest_sha256'])
        if asset.metadata['policy_sha256'] != item['policy_sha256']:
            raise ValueError('Source or opponent changed')
    from integrations.native_spatial_asset import canonical_json, sha256
    migration = inputs/'abi-migration'
    descriptors = [read(migration/name) for name in ('baseline.json', 'candidate.json')]
    if ({k for k in descriptors[0].keys() | descriptors[1].keys()
         if descriptors[0].get(k) != descriptors[1].get(k)} != {'direct_adapter_sha256'}):
        raise ValueError('Migration changed the native layout')
    abis = [sha256(canonical_json(d)) for d in descriptors]
    for arm, descriptor in zip(('baseline', 'fused'), descriptors, strict=True):
        if descriptor['direct_adapter_sha256'] != digest(inputs/'sources'/arm/'integrations/direct_spatial_optimization.py'):
            raise ValueError('Migration descriptor source differs')
    receipt = read(migration/'migration.json')
    if (receipt['proof_sha256'] != digest(migration/'proof.json')
            or receipt['proof'] != read(migration/'proof.json')
            or receipt['script_sha256'] != digest(migration/'migrate.py')
            or [receipt['proof']['old_abi_sha256'], receipt['proof']['new_abi_sha256']] != abis
            or receipt['proof']['baseline_descriptor_sha256'] != digest(migration/'baseline.json')
            or receipt['proof']['candidate_descriptor_sha256'] != digest(migration/'candidate.json')
            or [r['path'] for r in receipt['assets']] != ['assets/cold', 'bundles/cold']):
        raise ValueError('Migration receipt differs')
    for record in receipt['assets']:
        before_path, after_path = inputs/record['path'], inputs/'fused'/record['path']
        before = read(before_path/'asset.json')
        after = load_asset(after_path/'asset.json', manifest_sha256=record['new_manifest_sha256']).metadata
        if (digest(before_path/'asset.json') != record['old_manifest_sha256']
                or [before['abi_sha256'], after['abi_sha256']] != abis
                or {k:v for k,v in before.items() if k not in ('abi_sha256', 'provenance')}
                   != {k:v for k,v in after.items() if k not in ('abi_sha256', 'provenance')}):
            raise ValueError('Candidate migration changed weights, layout or objective')
        for name in ('policy.bin', 'policy.bin.learner', 'weights.npz'):
            if (before_path/name).exists() and digest(before_path/name) != digest(after_path/name):
                raise ValueError('Candidate migration changed parameter bytes')
    if output.exists():
        raise ValueError('Fresh diagnostic output required')
    output.mkdir(parents=True)
    write(output/'plan.json', plan)
    write(output/'classic-contract.json', validate_training_contract(build, config))
    return plan, build, config


def audit(stage, config, uuid):
    """Record sub-gate diagnostics without falsely labeling them qualifications."""
    from integrations.monitor_coworld_steady_interval import completed_epoch_times
    text = (stage/'run/console.log').read_text()
    times = completed_epoch_times(text)
    if set(times) != {1, 2, 3, 4} or read(stage/'run/completed.json')['trained_timesteps'] != 2097152:
        raise ValueError('Incomplete diagnostic budget')
    mixes = [json.loads(line.split('INITIAL_POSITION_MIX ', 1)[1])
             for line in text.splitlines() if 'INITIAL_POSITION_MIX ' in line]
    if not mixes or any(row != dict(games=4096, midgame=0, probability=0.0) for row in mixes):
        raise ValueError('Fresh reset audit differs')
    if 'DEVICE_ACTION_MASK_AUDIT actions=2097152 illegal=0' not in text:
        raise ValueError('Missing complete legal action audit')
    rewards = [json.loads(line.split(' ', 1)[1]) for line in text.splitlines()
               if line.startswith('DEVICE_REWARD_AUDIT ')]
    if not rewards or rewards[-1]['agent_steps'] != 2097152 or any(
            r[k] for r in rewards for k in ('nonfinite_rewards', 'native_clipped_rewards', 'native_clipped_terminal_rewards')):
        raise ValueError('Invalid reward audit')
    rotation_audit(stage, 0, 4)
    if digest(stage/'run/initial-policy.bin') != read(PLAN)['source_policy_sha256']:
        raise ValueError('Initializer changed')
    from integrations.native_startup_admission import verify_receipt
    admission = verify_receipt(stage/'run/training.json')
    if admission['initialization'] != 'fresh_optimizer' or admission['initial_learner_sha256'] is not None:
        raise ValueError('Optimizer was not freshly initialized')
    hardware = [json.loads(line) for line in (stage/'hardware.jsonl').read_text().splitlines()]
    if not hardware or any(row['gpu']['uuid'] != uuid for row in hardware):
        raise ValueError('GPU identity changed')
    population = read(stage/f"run/environments/{config['seed']}/spatial-opponent-population.json")
    build = read(stage/'build-config.json')['python_environment']['options']
    expected = [digest(Path(p)/'policy.bin') for p in build['frozen_bundles']]
    names = {'frozen_'+h[:12] for h in expected} | set(build['scripted_opponents'])
    if (population['frozen_policy_sha256'] != expected or population['opponent_weights'] != build['opponent_weights']
            or set(population['counts']) != names
            or not all(v['0'] > 0 and v['0'] == v['1'] for v in population['counts'].values())):
        raise ValueError('Opponent identity or seat coverage differs')
    seconds = times[4] - times[2]
    if seconds <= 0:
        raise ValueError('Invalid measured interval')
    report = dict(warmup_epochs=2, measured_steps=1048576, measured_seconds=seconds,
                  steady_sps=1048576/seconds, epoch_uptime=times, gpu_uuid=uuid,
                  qualification=False, opponent_counts=population['counts'],
                  sampled_peak_mib=max(float(row['gpu']['memory.used'].split()[0]) for row in hardware),
                  hardware_samples=len(hardware), geometry=dict(environments=4096, horizon=128, minibatch=8192, replay_ratio=.5, workers=8))
    write(stage/'diagnostic-audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from integrations.cuda_runtime_binding import configure
    configure()
    inputs, output = args.input.resolve(), args.output.resolve()
    plan, build, config = prepare(inputs, output)
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
    identity = verify_gpu_idle(visible_gpu_identity())
    write(output/'gpu-preflight.json', identity)
    sampler = read(inputs/'assets/cold/asset.json')['sampler']
    from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
    sampling = source_sampling_gate_report(inputs/'source-sampling')
    if (sampling['source_sha256'] != plan['source_policy_sha256']
            or sampling['opponent_sha256'] != plan['source_policy_sha256']
            or sampling['sampler'] != dict(full_action_temperature=1., route_half_weight=0., **sampler)
            or sampling['opponent_sampler'] != sampling['sampler']
            or sampling['match_seed'] != 11008101 or sampling['sample_seed'] != 11008103):
        raise ValueError('Historical sampling gate differs')
    for arm in ('baseline', 'fused'):
        source, out = inputs/'sources'/arm, output/arm
        write(out/'build-config.json', build)
        execute('launch_spatial_selfplay_training', ['build', '--config', out/'build-config.json', '--output', out/'build'],
                source=source, output=out, sampler=sampler, name='build', seconds=600)
        execute('audit_spatial_checkpoint_serving_parity', ['--bundle', (inputs/'fused/bundles/cold' if arm == 'fused' else inputs/'bundles/cold'), '--replay-root', inputs/'leader-root',
                '--factory-source', source/'integrations/generals_fabric.py', '--output', out/'source-parity.json'],
                source=source, output=out, sampler=sampler, name='source-parity', seconds=600)
        parity = read(out/'source-parity.json')
        from integrations.classic_contract import ENGINE_SHA256
        bundle = inputs/('fused/bundles/cold' if arm == 'fused' else 'bundles/cold')
        if (parity['public_states'] != 46 or parity['matching_top_actions'] != 46
                or parity['inference_backend'] != 'gpu'
                or not 0 <= parity['max_action_probability_difference'] <= 1e-5
                or not 0 <= parity['max_rollout_transform_difference'] <= 1e-5
                or parity['checkpoint_sha256'] != plan['source_policy_sha256']
                or parity['engine_sha256'] != ENGINE_SHA256
                or parity['factory_source_sha256'] != digest(source/'integrations/generals_fabric.py')
                or parity['serving_action_selection'] != sampler
                or parity['bundle_manifest_sha256'] != digest(bundle/'spatial-policy.json')):
            raise ValueError('Source serving parity failed')
    coordinator = Path(__file__).resolve().parents[1]
    for arm in ('baseline', 'fused'):
        arguments = ['--source', inputs/'sources'/arm, '--input', inputs,
                     '--asset', (inputs/'fused/assets/cold/asset.json' if arm == 'fused' else inputs/'assets/cold/asset.json'),
                     '--output', output/('dispatch-'+arm)]
        if arm == 'fused':
            arguments += ['--baseline', output/'dispatch-baseline']
        execute('audit_dispatch_fusion', arguments, source=coordinator, output=output,
                sampler=sampler, name='dispatch-'+arm, seconds=600)
    execute('audit_dispatch_fusion', ['--compare', output/'dispatch-baseline', output/'dispatch-fused',
            '--output', output/'dispatch-parity.json'], source=coordinator, output=output,
            sampler=sampler, name='dispatch-compare', seconds=120)
    parity = read(output/'dispatch-parity.json')
    if parity.get('passed') is not True or parity.get('backend') != 'gpu':
        raise ValueError('Production dispatch parity failed')
    for arm, expected in zip(('baseline', 'fused'), parity['admissions'], strict=True):
        receipt = output/('dispatch-'+arm)/'admission.json'
        admission = read(receipt)
        if (digest(receipt) != expected or admission['source_sha256'] != digest(inputs/'sources'/arm/'integrations/direct_spatial_optimization.py')
                or admission['audit_module_sha256'] != digest(coordinator/'integrations/audit_dispatch_fusion.py')):
            raise ValueError('Production dispatch audit source bindings differ')
    reports = []
    for index, arm in enumerate(ORDER):
        stage = output/f'{index+1}-{arm}'
        run_config = copy.deepcopy(config)
        if arm == 'fused':
            asset = inputs/'fused/assets/cold/asset.json'
            run_config['initialize'] = dict(asset=str(asset), manifest_sha256=digest(asset), restore_learner=False)
        write(stage/'config.json', run_config)
        write(stage/'build-config.json', read(output/arm/'build-config.json'))
        write(stage/'sampling-gate.json', sampling)
        execute('launch_spatial_selfplay_training', ['train', '--build', output/arm/'build', '--config', stage/'config.json',
                '--output', stage/'run'], source=inputs/'sources'/arm, output=stage, sampler=sampler, name='train',
                seconds=480, startup_seconds=300, training_config=stage/'config.json', diagnostic_profile=True)
        reports.append(dict(arm=arm, **audit(stage, run_config, identity['uuid'])))
    peaks = {arm: sum(r['sampled_peak_mib'] for r in reports if r['arm'] == arm)/2 for arm in ('baseline', 'fused')}
    write(output/'COMPLETED.json', dict(plan_sha256=digest(output/'plan.json'), stages=reports,
          mean_sampled_peak_mib=peaks, sampled_memory_saving_mib=peaks['baseline']-peaks['fused'],
          eligible_for_separate_qualification=all(r['steady_sps'] >= 30000 for r in reports if r['arm'] == 'fused'),
          qualified_for_long_training=False))


if __name__ == '__main__':
    main()

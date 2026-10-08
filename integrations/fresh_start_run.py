"""Fresh-game-only reset ablation with a new GPU qualification before continuation."""
import argparse
import copy
import json
import re
from pathlib import Path
from integrations.policy_trial import Trial, digest, read, write
from integrations.training_inputs import CONFIG, ASSETS
from integrations.policy_training_audit import qualification_audit

PLAN_PATH = Path(__file__).with_name('fresh_start_plan.json')
BINDINGS = {'control-audit.json', 'control-collection.json', 'control/candidate/build-config.json',
            'bundles/control/asset.json', 'bundles/control/policy.bin', 'hypothesis.json',
            'probe-audit.json', 'probe-collection.json', 'probe-context-seal.json'}


def bound_plan(inputs):
    plan = read(PLAN_PATH)
    if plan['status'] != 'sealed before launch' or set(plan['bindings']) != BINDINGS:
        raise ValueError('Fresh-start experiment is pending evidence bindings')
    for name, expected in plan['bindings'].items():
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected) or digest(inputs/name) != expected:
            raise ValueError('Unbound or changed evidence: ' + name)
    fixed = dict(training_seed=11007101, qualification_steps=4194304,
                 total_steps_including_qualification=33554432, evaluation_seed=17001101,
                 evaluation_sample_seed=17001103, bootstrap_seed=17001111,
                 evaluation_games_per_arm=4096, bootstrap_resamples=10000)
    if any(plan[key] != value for key, value in fixed.items()):
        raise ValueError('Preregistered experiment budget or seeds changed')
    if plan['runtime_bound'] != dict(provider_minutes=90, execution_minutes=88,
                                    aggregate_minutes_from_first_allocation=90, max_restarts=0):
        raise ValueError('Fixed allocation bound changed')
    return plan


def dispatch_admission(inputs, plan):
    """Require independently audited dispatch evidence and its exact frozen runtime."""
    import importlib.util
    audit = read(inputs/'probe-audit.json')
    collection = read(inputs/'probe-collection.json')
    if (audit['schema'] != 'generals-dispatch-probe-independent-audit-v1'
            or audit['job_id'] != plan['dispatch_probe_job_id'] or audit['status'] != 'succeeded'
            or audit['technical_success'] is not True
            or audit['eligible_for_separate_qualification'] is not True
            or audit['qualified_for_long_training'] is not False
            or audit['collection_sha256'] != digest(inputs/'probe-collection.json')
            or collection['complete'] is not True):
        raise ValueError('Dispatch diagnostic does not permit separate qualification')
    stages = audit['stages']
    if ([r['arm'] for r in stages] != ['baseline', 'fused', 'fused', 'baseline']
            or any(r['steady_sps'] < 30000 for r in stages if r['arm'] == 'fused')):
        raise ValueError('Both fused diagnostics must pass 30K')
    for name, expected in collection['files'].items():
        if digest(inputs/'probe'/name) != expected:
            raise ValueError('Retained probe artifact changed: ' + name)
    gate = read(inputs/'probe/dispatch-parity.json')
    if (gate['passed'] is not True or gate['backend'] != 'gpu'
            or gate['admissions'] != [digest(inputs/'probe'/('dispatch-'+arm)/'admission.json')
                                      for arm in ('baseline', 'fused')]):
        raise ValueError('Successful bound production GPU dispatch gate required')
    seal = read(inputs/'probe-context-seal.json')
    framework = Path(importlib.util.find_spec('metta_training.native_build').origin).parents[1]
    checked = {}
    prefix = 'input/sources/fused/'
    source_files = {str(p.relative_to(inputs/'source')) for p in (inputs/'source').rglob('*') if p.is_file()}
    if source_files != {name.removeprefix(prefix) for name in seal['files'] if name.startswith(prefix)}:
        raise ValueError('Frozen candidate source file set differs')
    for name, expected in seal['files'].items():
        if name.startswith('framework-source/'):
            actual = framework/Path(name).relative_to('framework-source')
        elif name.startswith(prefix):
            actual = inputs/'source'/name.removeprefix(prefix)
        elif name.startswith(('input/fused/assets/cold/', 'input/fused/bundles/cold/')):
            actual = inputs/Path(name).relative_to('input/fused')
        else:
            continue
        if digest(actual) != expected:
            raise ValueError('Audited dispatch execution or selected asset changed: ' + name)
        checked[name] = expected
    selected = {}
    for name in ('assets/cold/asset.json', 'bundles/cold/asset.json'):
        selected[name] = seal['files']['input/fused/'+name]
    receipt = read(inputs/'probe/dispatch-fused/admission.json')
    if (receipt['source_sha256'] != digest(inputs/'source/integrations/direct_spatial_optimization.py')
            or receipt['audit_module_sha256'] != seal['files']['coordinator/integrations/audit_dispatch_fusion.py']
            or receipt['policy_sha256'] != plan['source_policy_sha256']
            or receipt['abi_sha256'] != read(inputs/'assets/cold/asset.json')['abi_sha256']
            or receipt['teacher_ppo_coefficient'] != 1.0):
        raise ValueError('Audited candidate identity differs')
    return dict(probe_job_id=audit['job_id'], audit_sha256=digest(inputs/'probe-audit.json'),
                context_seal_sha256=digest(inputs/'probe-context-seal.json'), execution_files=checked,
                selected_assets=selected, qualification=False)


def intervals(stage, first, last):
    from integrations.monitor_coworld_steady_interval import completed_epoch_times
    text = (stage/'run/console.log').read_text()
    mixes = [json.loads(line.split('INITIAL_POSITION_MIX ', 1)[1])
             for line in text.splitlines() if 'INITIAL_POSITION_MIX ' in line]
    if not mixes or any(row != dict(games=4096, midgame=0, probability=0.0) for row in mixes):
        raise ValueError('Runtime fresh-game reset distribution differs')
    times = completed_epoch_times(text)
    if set(times) != set(range(first, last + 1)):
        raise ValueError('Completed epoch range differs')
    warm = first + 1
    windows = [(warm, last)] + [(end - 2, end) for end in range(warm + 2, last + 1)]
    rates = []
    for start, end in windows:
        seconds = times[end] - times[start]
        if seconds <= 0 or (end - start) * 524288 / seconds < 30000:
            raise ValueError('Full or rolling post-warmup throughput below 30K')
        rates.append(dict(first=start, last=end, seconds=seconds, sps=(end-start)*524288/seconds))
    write(stage/'throughput-intervals.json', rates)


def configs(inputs, admission):
    """Preserve reset and learner settings; bind the audited selected ABI metadata."""
    reference = read(CONFIG/'build-config.json')
    if read(inputs/'control/candidate/build-config.json') != reference:
        raise ValueError('Retained stateless control configuration differs')
    build = copy.deepcopy(reference)
    build['python_environment']['options']['coworld_position_probability'] = 0.0
    config = read(CONFIG/'training-config.json')
    config['initialize']['manifest_sha256'] = admission['selected_assets']['assets/cold/asset.json']
    if read(inputs/'build-config.json') != build or read(inputs/'config.json') != config:
        raise ValueError('Only the fixed fresh-start intervention is allowed')
    return build, config


def prepare(inputs, output):
    from integrations.native_spatial_asset import load_asset, training_contract
    from integrations.classic_contract import validate_training_contract
    plan = bound_plan(inputs)
    admission = dispatch_admission(inputs, plan)
    if output.exists():
        raise ValueError('Fresh output directory required')
    for name, expected in read(inputs/'source-manifest.json').items():
        if digest(inputs/name) != expected:
            raise ValueError('Staged input changed: ' + name)
    evidence = read(inputs/'control-audit.json')
    collection = read(inputs/'control-collection.json')
    if (evidence['job_id'] != 'job-mwvdb' or evidence['status'] != 'succeeded'
            or evidence['technical_success'] is not True or collection['complete'] is not True
            or evidence['collection_sha256'] != digest(inputs/'control-collection.json')):
        raise ValueError('Retained control lacks successful independent technical audit')
    for name, expected in collection['files'].items():
        if digest(inputs/'control'/name) != expected:
            raise ValueError('Retained control artifact changed: ' + name)
    for item in ASSETS:
        directory = inputs/Path(item['destination']).relative_to('/work/input')
        manifest = admission['selected_assets'].get(str(directory.relative_to(inputs)/'asset.json'), item['manifest_sha256'])
        asset = load_asset(directory/'asset.json', manifest_sha256=manifest)
        if asset.metadata['policy_sha256'] != item['policy_sha256']:
            raise ValueError('Fixed source or opponent identity changed')
    source = load_asset(inputs/'assets/cold/asset.json', manifest_sha256=admission['selected_assets']['assets/cold/asset.json'])
    control = load_asset(inputs/'bundles/control/asset.json', manifest_sha256=plan['bindings']['bundles/control/asset.json'])
    if (control.metadata['policy_sha256'] != plan['historical_control_policy_sha256']
            or control.metadata['sampler'] != source.metadata['sampler']
            or digest(inputs/'control/candidate/continuation/asset/policy.bin') != control.metadata['policy_sha256']):
        raise ValueError('Exact stateless control or sampler changed')
    build, config = configs(inputs, admission)
    if training_contract(build['python_environment']['options'], config['overrides']) != source.metadata['training_contract']:
        raise ValueError('Source objective changed')
    output.mkdir(parents=True)
    write(output/'plan.json', plan)
    write(output/'dispatch-admission.json', admission)
    write(output/'candidate/classic-contract.json', validate_training_contract(build, config))
    write(output/'candidate/build-config.json', build)
    write(output/'candidate/qualification/config.json', config)
    write(output/'source-binding.json', dict(source_policy_sha256=source.metadata['policy_sha256'],
          parameter_count=source.metadata['parameter_count'], input_manifest_sha256=digest(inputs/'source-manifest.json')))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bound_plan(args.input)
    from integrations.cuda_runtime_binding import configure
    configure()
    plan = prepare(args.input.resolve(), args.output.resolve())
    trial = Trial(args.input, args.output, plan=plan)
    trial.smoke()
    trial.build()
    trial.qualify()
    # Full epoch2->8 gate supplements the live rolling gate before any long training.
    qualification_audit(args.output, destination=args.output/'qualification-audit.json')
    intervals(args.output/'candidate/qualification', 1, 8)
    trial.continue_training()
    from integrations.policy_training_audit import finish
    finish(args.output)
    intervals(args.output/'candidate/continuation', 9, 64)
    trial.evaluate()
    write(args.output/'COMPLETED.json', dict(plan_sha256=digest(PLAN_PATH),
          selected=read(args.output/'selection.json')['selected'], requires_fresh_confirmation=True))


if __name__ == '__main__':
    main()

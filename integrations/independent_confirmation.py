"""One gated frozen confirmation using the existing population evaluator."""
import argparse
import os
from pathlib import Path
import re
import sys

from integrations.row_rotation_trial import digest, read, write, selected

PLAN_PATH = Path(__file__).with_name('independent_confirmation_plan.json')
BINDINGS = {'development-audit.json', 'development-collection.json', 'development-context-seal.json',
            'development/candidate/build/build.json', 'development/candidate/continuation/bundle/asset.json'}


def plan_for(inputs):
    plan = read(PLAN_PATH)
    if plan['status'] != 'sealed after positive independent development audit':
        raise ValueError('Confirmation awaits positive independently audited development')
    if (plan['development_job'] != 'job-mwvdb' or set(plan['bindings']) != BINDINGS
            or plan['source_policy_sha256'] != 'f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14'
            or plan['control_policy_sha256'] != 'f4b185ef31755097aea0435266d68f8a1c8aa10330bb03d801cf1fcb0f43e65b'):
        raise ValueError('Development bindings differ')
    for name, expected in plan['bindings'].items():
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Pending development binding: ' + name)
        if digest(inputs / name) != expected:
            raise ValueError('Development evidence changed: ' + name)
    if not isinstance(plan['candidate_policy_sha256'], str) or not re.fullmatch('[a-f0-9]{64}', plan['candidate_policy_sha256']):
        raise ValueError('Final candidate identity is pending')
    if tuple(plan[k] for k in ('evaluation_seed', 'evaluation_sample_seed', 'bootstrap_seed',
                               'games_per_arm', 'bootstrap_resamples')) != (16001101, 16001103, 16001111, 4096, 10000):
        raise ValueError('Preregistered confirmation panel changed')
    if plan['bounds'] != dict(execution_minutes=48, provider_minutes=50, aggregate_minutes_from_first_allocation=50, max_restarts=0):
        raise ValueError('Confirmation bounds changed')
    return plan


def admit(inputs):
    from integrations.analyze_spatial_population_pair import compare
    from integrations.native_spatial_asset import load_asset
    plan = plan_for(inputs)
    audit, collection = read(inputs / 'development-audit.json'), read(inputs / 'development-collection.json')
    if (audit['schema'] != 'generals-stateless-continuation-independent-audit-v1'
            or audit['job_id'] != 'job-mwvdb' or audit['status'] != 'succeeded'
            or audit['technical_success'] is not True or audit['selected'] is not True
            or audit['collection_sha256'] != digest(inputs / 'development-collection.json')
            or collection['complete'] is not True):
        raise ValueError('Development did not qualify confirmation')
    dev = inputs / 'development'
    actual = {str(p.relative_to(dev)): digest(p) for p in dev.rglob('*') if p.is_file()}
    if actual != collection['files']:
        raise ValueError('Development collection differs')
    # Recompute the existing development decision from its retained raw arrays.
    comparisons = [compare(dev / ('heldout-' + arm), dev / 'heldout-candidate',
                           seed=14001111, resamples=10000) for arm in ('source', 'control')]
    if any(report['baseline_checkpoint_sha256'] != plan[arm + '_policy_sha256']
           or report['candidate_checkpoint_sha256'] != plan['candidate_policy_sha256']
           for arm, report in zip(('source', 'control'), comparisons)):
        raise ValueError('Development comparisons identify different actors')
    if not selected(comparisons):
        raise ValueError('Raw development outcomes do not pass both comparisons')
    seal = read(inputs / 'development-context-seal.json')['files']
    for name, expected in seal.items():
        if name.startswith('input/bundles/') and digest(inputs.parent / name) != expected:
            raise ValueError('Frozen population or reference bundle changed: ' + name)
    bundles = {'source': inputs / 'bundles/cold', 'control': inputs / 'bundles/control',
               'candidate': dev / 'candidate/continuation/bundle'}
    actors = {}
    for name, bundle in bundles.items():
        manifest = (plan['bindings']['development/candidate/continuation/bundle/asset.json']
                    if name == 'candidate' else seal['input/' + str(bundle.relative_to(inputs)) + '/asset.json'])
        asset = load_asset(bundle / 'asset.json', manifest_sha256=manifest)
        if asset.metadata['policy_sha256'] != plan[name + '_policy_sha256']:
            raise ValueError('Frozen actor identity changed: ' + name)
        actors[name] = asset
    if any(a.metadata['sampler'] != actors['source'].metadata['sampler'] for a in actors.values()):
        raise ValueError('Frozen sampler changed')
    build = dev / 'candidate/build/build.json'
    population = read(build)['config']['python_environment']['options']
    for directory in population['frozen_bundles']:
        relative = Path(directory).relative_to('/work/input')
        actors[str(relative)] = load_asset(inputs / relative / 'asset.json',
                                          manifest_sha256=seal['input/' + str(relative) + '/asset.json'])
    seeds = {plan[k] for k in ('evaluation_seed', 'evaluation_sample_seed', 'bootstrap_seed')}
    if any(seeds & set(a.metadata['training_seeds']) for a in actors.values()):
        raise ValueError('Confirmation seeds overlap a training lineage')
    # Preserve actual evaluation/game code; only this new coordinator may differ.
    for name, expected in seal.items():
        if (name.startswith('input/source/generals/') and Path(name).suffix in ('.py', '.cpp')
                or name.startswith('input/source/integrations/') and Path(name).suffix in ('.py', '.cpp')
                and Path(name).name not in ('stateless_continuation_run.py', 'stateless_continuation_entrypoint.py')):
            if digest(inputs.parent / name) != expected:
                raise ValueError('Development execution source changed: ' + name)
    return plan, bundles, build


def evaluate(inputs, output):
    from integrations.policy_execution import execute
    plan, bundles, build = admit(inputs)
    if output.exists():
        raise ValueError('Confirmation requires a fresh output directory')
    output.mkdir()
    write(output / 'plan.json', plan)
    for arm, bundle in bundles.items():
        execute('evaluate_spatial_population', ['--bundle', bundle, '--population-build', build,
                '--games', 4096, '--pool-size', 4096, '--seed', 16001101, '--sample-seed', 16001103,
                '--destination-audit', '--output', output / ('heldout-' + arm)],
                source=inputs / 'source', output=output, sampler=read(bundle / 'asset.json')['sampler'],
                name='evaluate-' + arm, seconds=900)
    from integrations.analyze_spatial_population_pair import compare
    reports = []
    for arm in ('source', 'control'):
        report = compare(output / ('heldout-' + arm), output / 'heldout-candidate', seed=16001111, resamples=10000)
        write(output / ('paired-' + arm + '-candidate.json'), report)
        reports.append(report)
    write(output / 'selection.json', dict(selected=selected(reports), comparisons=reports,
          hosted_qualification_required=True, development_audit_sha256=digest(inputs / 'development-audit.json')))


def completion(output):
    return dict(selected=read(output / 'selection.json')['selected'],
                selection_sha256=digest(output / 'selection.json'), hosted_qualification_required=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('/work/input'))
    parser.add_argument('--output', type=Path, default=Path('/work/results/generals'))
    parser.add_argument('--evaluate', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.input, args.output)
    else:
        from integrations.bounded_policy_entrypoint import run
        run([([sys.executable, '-u', '-m', 'integrations.independent_confirmation', '--evaluate',
               '--input', str(args.input), '--output', str(args.output)], dict(os.environ))],
            args.output, Path(os.environ.get('GMN_OUTPUT_DIR', '/output')), 48 * 60,
            kind='independent-confirmation', completion=completion)


if __name__ == '__main__':
    main()

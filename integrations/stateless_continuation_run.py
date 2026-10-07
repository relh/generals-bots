"""Conditional continuation of the exact qualified stateless learner."""
import argparse
import re
import shutil
import signal
from pathlib import Path

from integrations import row_rotation_trial as trial_module
from integrations.row_rotation_trial import Trial, digest, read, write

PLAN_PATH = Path(__file__).with_name('stateless_continuation_plan.json')
CHECKPOINT = 'run/checkpoints/metta_generals/run/0000000004194304.bin'


def bound_plan(inputs):
    plan = read(PLAN_PATH)
    if (plan['status'] != 'sealed after qualifying terminal audit'
            or not isinstance(plan['qualification_job_id'], str)
            or not re.fullmatch(r'job-[a-z0-9]+', plan['qualification_job_id'])):
        raise ValueError('Continuation is pending a reviewed qualifying terminal audit')
    for name, expected in plan['bindings'].items():
        if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Unbound qualification input: ' + name)
        if digest(inputs / name) != expected:
            raise ValueError('Qualification input changed: ' + name)
    for key in ('training_seed', 'evaluation_seed', 'evaluation_sample_seed', 'bootstrap_seed',
                'evaluation_games_per_arm', 'bootstrap_resamples'):
        if plan[key] != trial_module.PLAN[key]:
            raise ValueError('Reused evaluator contract differs: ' + key)
    if tuple(plan[k] for k in ('starting_run_steps', 'starting_epoch', 'ending_run_steps',
                              'ending_epoch', 'incremental_steps')) != (4194304, 8, 33554432, 64, 29360128):
        raise ValueError('Fixed cumulative clock changed')
    return plan


def clock(path, words, steps, epoch):
    from integrations.learner_checkpoint import LearnerCheckpoint
    state = LearnerCheckpoint.from_bytes(path.read_bytes(), words)
    if state.agent_steps != steps or state.epoch != epoch:
        raise ValueError('Authenticated learner cumulative clock differs')


def prepare(inputs, output):
    from integrations.native_spatial_asset import load_asset
    from integrations.stateless_qualification_run import PLAN as qualification_plan
    plan = bound_plan(inputs)
    if output.exists():
        raise ValueError('Continuation requires a fresh output directory')
    audit = read(inputs / 'qualification-audit.json')
    collection = read(inputs / 'qualification-collection.json')
    retained = inputs / 'qualification'
    if (audit['schema'] != 'generals-stateless-qualification-independent-audit-v1'
            or audit['job_id'] != plan['qualification_job_id'] or audit['status'] != 'succeeded'
            or audit['qualified'] is not True or audit['steady_sps'] < 30000
            or audit['collection_sha256'] != digest(inputs / 'qualification-collection.json')
            or collection['complete'] is not True):
        raise ValueError('Terminal evidence does not qualify continuation')
    for name, expected in collection['files'].items():
        if digest(retained / name) != expected:
            raise ValueError('Retained qualification file changed: ' + name)
    for name, expected in read(inputs / 'source-manifest.json').items():
        if digest(inputs / name) != expected:
            raise ValueError('Sealed continuation input changed: ' + name)
    for item in qualification_plan['assets']:
        directory = inputs / Path(item['destination']).relative_to('/work/input')
        asset = load_asset(directory / 'asset.json', manifest_sha256=item['manifest_sha256'])
        if asset.metadata['policy_sha256'] != item['policy_sha256']:
            raise ValueError('Migrated fixed asset changed')
    control = load_asset(inputs / 'bundles/control/asset.json',
                         manifest_sha256=plan['historical_control']['manifest_sha256'])
    source = load_asset(inputs / 'assets/cold/asset.json',
                        manifest_sha256=qualification_plan['assets'][0]['manifest_sha256'])
    if (control.metadata['policy_sha256'] != plan['historical_control']['policy_sha256']
            or control.metadata['sampler'] != source.metadata['sampler']):
        raise ValueError('Historical control policy or sampler changed')
    q = retained / 'candidate/qualification' 
    asset = load_asset(q / 'asset/asset.json', manifest_sha256=digest(q / 'asset/asset.json'))
    checkpoint = q / CHECKPOINT
    if digest(checkpoint) != digest(q / 'asset/policy.bin'):
        raise ValueError('Published qualified weights differ from checkpoint')
    learner = checkpoint.with_name(checkpoint.name + '.learner')
    if digest(learner) != digest(q / 'asset/policy.bin.learner'):
        raise ValueError('Published qualified optimizer differs from checkpoint')
    clock(learner, asset.metadata['parameter_count'], 4194304, 8)
    from integrations.stateless_qualification_run import CONFIG
    config = read(q / 'config.json')
    if (config != read(CONFIG / 'training-config.json')
            or read(retained / 'candidate/build-config.json') != read(CONFIG / 'build-config.json')
            or asset.metadata['sampler'] != source.metadata['sampler']):
        raise ValueError('Qualified training contract changed')
    if config['seed'] != 11007101 or config['total_timesteps'] != 4194304:
        raise ValueError('Qualification seed or run budget differs')
    output.mkdir()
    shutil.copytree(q, output / 'candidate/qualification')
    write(output / 'candidate/build-config.json', read(retained / 'candidate/build-config.json'))
    write(output / 'plan.json', plan)
    marker = {p: digest(q / p) for p in ('training-audit.json', 'asset/asset.json', 'serving-parity.json')}
    write(output / 'qualified.json', {'candidate': marker})
    config['total_timesteps'] = 33554432
    config['initialize'] = dict(asset=str(output / 'candidate/qualification/asset/asset.json'),
                               manifest_sha256=digest(q / 'asset/asset.json'), restore_learner=True)
    write(output / 'candidate/continuation/config.json', config)
    write(output / 'qualification-binding.json', dict(audit_sha256=digest(inputs / 'qualification-audit.json'),
          collection_sha256=digest(inputs / 'qualification-collection.json'), checkpoint_sha256=digest(checkpoint),
          learner_sha256=digest(learner), qualification_job_id=plan['qualification_job_id']))


def finish(output):
    stage = output / 'candidate/continuation'
    asset = read(stage / 'asset/asset.json')
    clock(stage / 'asset/policy.bin.learner', asset['parameter_count'], 33554432, 64)
    audit = read(stage / 'training-audit.json')
    if (audit['starting_agent_steps'] != 4194304 or audit['ending_agent_steps'] != 33554432
            or audit['environment_steps'] != 29360128 or audit['steady_sps'] < 30000):
        raise ValueError('Continuation clock or strict throughput gate differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = bound_plan(args.input)
    # One process bounds build, continuation and all three evaluation panels.
    def expired(*_):
        raise TimeoutError('Continuation execution budget exhausted')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(plan['runtime_bound']['execution_minutes'] * 60)
    from integrations.cuda_runtime_binding import configure
    configure()
    prepare(args.input.resolve(), args.output.resolve())
    trial = Trial(args.input, args.output)
    trial.smoke()
    trial.build()
    trial.continue_training()  # Existing exact initializer gate and strict live 30K guard.
    finish(args.output)
    trial.evaluate()
    write(args.output / 'COMPLETED.json', dict(plan_sha256=digest(PLAN_PATH),
          selected=read(args.output / 'selection.json')['selected'], requires_fresh_confirmation=True))


if __name__ == '__main__':
    main()

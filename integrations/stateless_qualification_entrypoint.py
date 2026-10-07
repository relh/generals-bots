"""Finite stateless qualification with retained artifacts and no continuation."""
import os
from pathlib import Path
import sys
from integrations.bounded_policy_entrypoint import run
from integrations.row_rotation_trial import digest, read
from integrations.stateless_qualification_run import PLAN


def completion(results):
    qualification = read(results / 'qualified.json')
    if qualification['qualified'] is not True:
        raise ValueError('Qualification did not pass')
    return dict(qualified=True, continuation_authorized=False,
                qualification_sha256=digest(results / 'qualified.json'))


def main():
    seconds = 60 * PLAN['bounds']['internal_minutes']
    if seconds >= 60 * PLAN['bounds']['provider_minutes']:
        raise ValueError('Execution limit must reserve time for artifact collection')
    results = Path('/work/results/generals')
    commands = []
    for phase in ('prepare', 'smoke', 'build', 'qualify'):
        env = dict(os.environ)
        if phase == 'prepare':
            env['JAX_PLATFORMS'] = 'cpu'
        commands.append(([sys.executable, '-u', '-m', 'integrations.stateless_qualification_run',
                          phase, '--input', '/work/input', '--output', str(results)], env))
    run(commands, results, Path(os.environ.get('GMN_OUTPUT_DIR', '/output')), seconds,
        kind='stateless-qualification', completion=completion)


if __name__ == '__main__':
    main()

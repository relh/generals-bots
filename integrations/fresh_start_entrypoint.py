"""Finite GMN fresh-start experiment with retained artifacts, including failures."""
import os
from pathlib import Path
import sys
from integrations.bounded_policy_entrypoint import run
from integrations.policy_trial import digest, read


def completion(results):
    marker = read(results / 'COMPLETED.json')
    if marker['plan_sha256'] != digest(results / 'plan.json'):
        raise ValueError('Coordinator completion plan differs')
    return dict(selected=marker['selected'], selection_sha256=digest(results / 'selection.json'),
                requires_fresh_confirmation=True)


def main():
    results = Path('/work/results/generals')
    command = [sys.executable, '-u', '-m', 'integrations.fresh_start_run',
               '--input', '/work/input', '--output', str(results)]
    run([(command, dict(os.environ))], results, Path(os.environ.get('GMN_OUTPUT_DIR', '/output')),
        88 * 60, kind='fresh-start', completion=completion)


if __name__ == '__main__':
    main()

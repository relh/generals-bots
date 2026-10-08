"""Bounded GPU dispatch diagnostic with retained partial outputs."""
import os
from pathlib import Path
import sys
from integrations.bounded_policy_entrypoint import run
from integrations.policy_trial import read, digest


def completion(results):
    report = read(results/'COMPLETED.json')
    if report['plan_sha256'] != digest(results/'plan.json') or report['qualified_for_long_training'] is not False:
        raise ValueError('Diagnostic completion binding differs')
    return {key: value for key, value in report.items() if key != 'plan_sha256'}


def main():
    results = Path('/work/results/generals')
    command = [sys.executable, '-u', '-m', 'integrations.dispatch_probe_run', '--input', '/work/input', '--output', str(results)]
    run([(command, dict(os.environ))], results, Path(os.environ.get('GMN_OUTPUT_DIR', '/output')),
        58*60, kind='dispatch-probe', completion=completion)


if __name__ == '__main__':
    main()

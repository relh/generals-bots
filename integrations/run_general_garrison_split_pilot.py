"""Three-arm, same-weight Classic split diagnostic for a sealed GPU context.

Expected inputs: /opt/generals-context/input/{source,bundles,population-build.json,plan.json}.
The context builder seals those files; this runner writes only to GMN_OUTPUT_DIR.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil

from integrations.cuda_runtime_binding import configure

configure()

from integrations.policy_execution import execute  # noqa: E402
from integrations.classic_contract import verify_engine  # noqa: E402
from integrations.slurm_s3_job import verify_gpu_idle, visible_gpu_identity  # noqa: E402
from integrations.spatial_policy_bundle import SpatialPlayerPolicy  # noqa: E402


def main():
    root = Path('/opt/generals-context')
    inputs = root / 'input'
    sealed = json.loads((root / 'seal.json').read_text())
    actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in root.rglob('*') if path.is_file() and path.name != 'seal.json'}
    if actual != sealed:
        raise ValueError('GPU context differs from sealed inputs')
    plan = json.loads((inputs / 'plan.json').read_text())
    if verify_engine() != plan['engine_sha256']:
        raise ValueError('Classic engine differs from preregistration')
    if (plan['biases'] != [0, 2, 4] or plan['games_per_arm'] != 4096
            or plan['pool_size'] != 4096 or plan['map_seed'] != 9874001
            or plan['sample_seed'] != 9874003 or plan['expected_opponents'] != 13):
        raise ValueError('Split pilot differs from preregistered arms and seeds')
    source, bundle = inputs / 'source', inputs / 'bundles/penalty8'
    if hashlib.sha256((bundle / 'spatial-policy.json').read_bytes()).hexdigest() != plan['bundle_manifest_sha256']:
        raise ValueError('Pilot bundle manifest differs from preregistration')
    policy = SpatialPlayerPolicy(bundle)
    if policy.asset.metadata['policy_sha256'] != plan['checkpoint_sha256']:
        raise ValueError('Pilot bundle checkpoint differs from preregistration')
    if policy.doomed_attack_route_penalty != 8:
        raise ValueError('Pilot requires the frozen penalty-eight serving bundle')
    build = inputs / 'population-build.json'
    if hashlib.sha256(build.read_bytes()).hexdigest() != plan['population_build_sha256']:
        raise ValueError('Opponent population build differs from preregistration')
    identity = verify_gpu_idle(visible_gpu_identity())
    output = Path(os.environ['GMN_OUTPUT_DIR']) / 'generals'
    output.mkdir()
    work = Path('/work')
    work.mkdir(exist_ok=True)
    (work / 'input').symlink_to(inputs, target_is_directory=True)
    for name in ('tmp', 'cache', 'jax-cache', 'fabric-verify'):
        (work / name).mkdir()
    (output / 'gpu-preflight.json').write_text(json.dumps(identity, indent=2) + '\n')
    shutil.copy(inputs / 'plan.json', output / 'plan.json')
    sampler = policy.asset.metadata['sampler']
    for bias in plan['biases']:
        arm = f'bias{bias}'
        execute('evaluate_spatial_population', [
            '--bundle', bundle, '--population-build', build,
            '--games', plan['games_per_arm'], '--pool-size', plan['pool_size'],
            '--seed', plan['map_seed'], '--sample-seed', plan['sample_seed'],
            '--general-garrison-split-bias', bias, '--output', output / arm,
        ], source=source, output=output, sampler=sampler, name=arm, seconds=1800)
        report = json.loads((output / arm / 'evaluation.json').read_text())
        if (not report['coworld_classic_rules'] or report['games'] != 4096
                or report['checkpoint_sha256'] != plan['checkpoint_sha256']
                or report['general_garrison_split_bias'] != bias
                or len(report['by_opponent_and_seat']) != 13):
            raise ValueError(f'Pilot arm {arm} differs from preregistration')
    for bias in (2, 4):
        execute('analyze_spatial_population_pair', [
            '--baseline', output / 'bias0', '--candidate', output / f'bias{bias}',
            '--output', output / f'comparison-bias{bias}.json',
        ], source=source, output=output, sampler={}, name=f'compare-bias{bias}', seconds=90)
    (output / 'COMPLETED.json').write_text(json.dumps({
        'development_only': True, 'checkpoint_sha256': plan['checkpoint_sha256'],
        'games_per_arm': 4096, 'arms': plan['biases'],
        'initial_state_pairing_verified': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Run the sealed, bounded three-arm Classic capital-threat diagnostic on one GPU."""

import hashlib
import json
import os
from pathlib import Path
import shutil

from integrations.cuda_runtime_binding import configure

configure()

from integrations.classic_contract import verify_engine  # noqa: E402
from integrations.policy_execution import execute  # noqa: E402
from integrations.slurm_s3_job import verify_gpu_idle, visible_gpu_identity  # noqa: E402
from integrations.spatial_policy_bundle import SpatialPlayerPolicy  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('/opt/generals-context')
    inputs = root / 'input'
    sealed = json.loads((root / 'seal.json').read_text())
    actual = {str(path.relative_to(root)): sha256(path)
              for path in root.rglob('*') if path.is_file() and path.name != 'seal.json'}
    if actual != sealed:
        raise ValueError('GPU context differs from sealed inputs')
    plan = json.loads((inputs / 'plan.json').read_text())
    if (plan['bonuses'] != [0, 2, 4] or plan['games_per_arm'] != 4096
            or plan['pool_size'] != 4096 or plan['map_seed'] != 9774001
            or plan['sample_seed'] != 9774003 or plan['bootstrap_seed'] != 9774011
            or plan['expected_opponents'] != 13):
        raise ValueError('Capital pilot differs from preregistered arms and seeds')
    source, bundle = inputs / 'source', inputs / 'bundle'
    if verify_engine() != plan['engine_sha256']:
        raise ValueError('Classic engine differs from preregistration')
    if sha256(source / 'docs/policy/capital-threat-gather-pilot.md') != plan['plan_document_sha256']:
        raise ValueError('Preregistered plan document differs')
    if sha256(bundle / 'spatial-policy.json') != plan['bundle_manifest_sha256']:
        raise ValueError('Frozen serving bundle differs from preregistration')
    policy = SpatialPlayerPolicy(bundle)
    if (policy.asset.metadata['policy_sha256'] != plan['checkpoint_sha256']
            or policy.doomed_attack_route_penalty != 8):
        raise ValueError('Expected unchanged penalty-eight source policy')
    build = inputs / 'population-build.json'
    if sha256(build) != plan['population_build_sha256']:
        raise ValueError('Opponent population differs from preregistration')
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
    for bonus in plan['bonuses']:
        arm = f'bonus{bonus}'
        execute('evaluate_spatial_population', [
            '--bundle', bundle, '--population-build', build,
            '--games', 4096, '--pool-size', 4096,
            '--seed', 9774001, '--sample-seed', 9774003,
            '--capital-threat-gather-bonus', bonus, '--destination-audit',
            '--output', output / arm,
        ], source=source, output=output, sampler=sampler, name=arm, seconds=1800)
        report = json.loads((output / arm / 'evaluation.json').read_text())
        if (not report['coworld_classic_rules'] or report['games'] != 4096
                or report['checkpoint_sha256'] != plan['checkpoint_sha256']
                or report['action_selection']['capital_threat_gather_bonus'] != bonus
                or len(report['by_opponent_and_seat']) != 13
                or 'destination_audit' not in report):
            raise ValueError(f'Pilot arm {arm} differs from preregistration')
    eligible = []
    for bonus in (2, 4):
        name = f'comparison-bonus{bonus}.json'
        execute('analyze_spatial_population_pair', [
            '--baseline', output / 'bonus0', '--candidate', output / f'bonus{bonus}',
            '--seed', 9774011, '--bootstrap-resamples', 10000,
            '--output', output / name,
        ], source=source, output=output, sampler={}, name=f'compare-bonus{bonus}', seconds=90)
        result = json.loads((output / name).read_text())
        strata = [row for opponent in result['by_opponent_and_seat'].values()
                  for row in opponent.values() if row['games'] >= 100]
        if (result['initial_state_cluster_ci95'][0] > 0
                and all(row['paired_signed_score_delta'] >= -0.15 for row in strata)):
            eligible.append((result['paired_signed_score_delta'], bonus))
    selected = max(eligible)[1] if eligible else 0
    (output / 'decision.json').write_text(json.dumps({
        'selected_bonus_for_independent_confirmation': selected,
        'eligible_candidates': [bonus for _, bonus in eligible],
        'confirmation_map_seed': 9775001, 'confirmation_sample_seed': 9775003,
        'development_only': True,
    }, indent=2) + '\n')
    (output / 'COMPLETED.json').write_text(json.dumps({
        'development_only': True, 'checkpoint_sha256': plan['checkpoint_sha256'],
        'games_per_arm': 4096, 'arms': plan['bonuses'],
        'initial_state_pairing_verified': True,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()

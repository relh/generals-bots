"""Stage and run a matched Classic PPO comparison of opponent weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import stat
import subprocess
from copy import deepcopy
from pathlib import Path

from integrations.classic_contract import validate_training_contract

PLAN = Path(__file__).resolve().parents[1] / 'docs/policy/hard-opponent-weighting.json'
ARMS = ('control', 'treatment')
QUALIFIED_PROBE_JOB_ID = 'job-izcgj'
QUALIFIED_PROBE_INPUT_SEAL_SHA256 = 'c922911720128b7a2d1f5c60dae88d2eae17a5c0643ba7ff45cdfc27cb6ed91b'
QUALIFIED_PROBE_SHA256 = '6fd0a28c23fdfe252ae7b86fc60b08df0b5e595bc504d0d63f39d341fda0df44'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob('*'))
            if path.is_file() and path != root / 'seal.json'}


def load_plan(path: Path = PLAN) -> dict:
    plan = json.loads(path.read_text())
    if (plan['experiment'] != 'hard-opponent-weighting-v1'
            or plan['control_weights'] != [1, 2, 1, 1, 1, 4, 5, 5, 11, 17, 3, 4, 16]
            or plan['treatment_weights'] != [1, 2, 1, 1, 1, 4, 5, 5, 11, 34, 3, 4, 32]
            or plan['training']['parallel_games'] != 4096
            or plan['training']['horizon'] != 128
            or plan['training']['minibatch'] != 8192
            or plan['training']['steps_per_arm'] != 16_777_216
            or plan['required_treatment_probe']['steps'] != 4_194_304
            or plan['required_treatment_probe']['minimum_steady_sps'] != 30_000
            or plan['development']['games_per_arm'] != 4096):
        raise ValueError('Hard-opponent preregistration differs from the frozen experiment')
    return plan


def derive_pair(build: dict, run: dict, plan: dict) -> tuple[dict, dict, dict]:
    """Change geometry equally in both arms and opponent weights only in treatment."""
    options = build['python_environment']['options']
    if (options['opponent_weights'] != plan['control_weights']
            or len(options['frozen_bundles']) != 10
            or options['scripted_opponents'] != plan['opponent_order'][10:]
            or options['horizon'] != 2000 or options['balance_opponent_sides'] is not True
            or options['terminal_reward_mode'] != 'win_only'
            or options['coworld_position_probability'] != 0.25
            or options['shaping_gamma'] != run['overrides']['train.gamma']):
        raise ValueError('Original Classic opponent or reward configuration differs')
    if (run['initialize']['asset'] != '/work/input/assets/cold/asset.json'
            or run['initialize']['restore_learner'] is not False):
        raise ValueError('Both arms require the same source actor and fresh optimizer')
    control, treatment, tuned = deepcopy(build), deepcopy(build), deepcopy(run)
    geometry = plan['training']
    for arm in (control, treatment):
        arm['python_environment']['options']['parallel_games'] = geometry['parallel_games']
        arm['python_environment']['spec']['agents'] = geometry['parallel_games']
    treatment['python_environment']['options']['opponent_weights'] = plan['treatment_weights']
    tuned['overrides']['vec.total_agents'] = geometry['parallel_games']
    tuned['overrides']['train.horizon'] = geometry['horizon']
    tuned['overrides']['train.minibatch_size'] = geometry['minibatch']
    tuned['overrides']['train.replay_ratio'] = geometry['replay_ratio']
    tuned['total_timesteps'] = geometry['steps_per_arm']
    tuned['seed'] = geometry['seed']
    expected = deepcopy(control)
    expected['python_environment']['options']['opponent_weights'] = plan['treatment_weights']
    if treatment != expected:
        raise ValueError('Treatment differs beyond opponent weights')
    for arm in (control, treatment):
        contract = json.loads(json.dumps(validate_training_contract(arm, tuned)))
        if (contract['engine_sha256'] != plan['engine_sha256']
                or contract['training_geometry']['steps_per_epoch'] != 524_288
                or contract['shaping_gamma'] != contract['learner_gamma']):
            raise ValueError('Classic source, discount, or geometry differs')
    return control, treatment, tuned


def verify_source(input_root: Path, plan: dict) -> None:
    asset = input_root / 'assets/cold/asset.json'
    cold = json.loads(asset.read_text())
    serving = json.loads((input_root / 'bundles/cold/asset.json').read_text())
    migration = plan['migrated_source']
    proof_path = input_root / 'source-migration-proof.json'
    proof = json.loads(proof_path.read_text())
    if (digest(input_root / 'assets/cold/policy.bin') != plan['source_checkpoint_sha256']
            or digest(input_root / 'bundles/cold/policy.bin') != plan['source_checkpoint_sha256']
            or digest(input_root / 'assets/cold/policy.bin.learner') != plan['source_learner_sha256']
            or cold['learner_sha256'] != plan['source_learner_sha256']
            or digest(asset) != migration['asset_manifest_sha256']
            or cold['model_sha256'] != migration['model_sha256']
            or serving['model_sha256'] != cold['model_sha256']
            or serving['policy_sha256'] != cold['policy_sha256']
            or serving['sampler'] != cold['sampler']
            or cold['abi_sha256'] != proof['abi_sha256']
            or digest(proof_path) != migration['proof_sha256']
            or proof['new_model_sha256'] != migration['model_sha256']
            or proof['policy_sha256'] != plan['source_checkpoint_sha256']
            or proof['reinforcement_learning_steps_added'] != 0
            or digest(asset) != json.loads((input_root / 'config.json').read_text())['initialize']['manifest_sha256']
            or cold['factory_source_sha256'] != digest(input_root / 'source/integrations/generals_fabric.py')):
        raise ValueError('Migrated source initializer, model identity, bundle, or factory differs')
    build = json.loads((input_root / 'build-config.json').read_text())
    frozen = build['python_environment']['options']['frozen_bundles']
    checksums = [digest(input_root / Path(path).relative_to('/work/input') / 'policy.bin') for path in frozen]
    if (['frozen_' + value[:12] for value in checksums] != plan['opponent_order'][:10]
            or checksums[0] == plan['source_checkpoint_sha256']):
        raise ValueError('Original frozen pool changed or contains a source mirror')


def prepare(source_input: Path, repository: Path, output: Path, plan_path: Path = PLAN) -> dict:
    if output.exists():
        raise ValueError('Use a fresh staged-input directory')
    source_input, repository = source_input.resolve(strict=True), repository.resolve(strict=True)
    plan = load_plan(plan_path)
    if source_input != Path(plan['source_input']).resolve(strict=True):
        raise ValueError('Source input path differs from preregistration')
    output.mkdir(parents=True)
    for name in ('assets/cold', 'bundles/cold', 'bundles/frozen', 'curriculum',
                 'puffer.git', 'raylib-5.5_linux_amd64'):
        shutil.copytree(source_input / name, output / name,
                        ignore=shutil.ignore_patterns('._*', '.DS_Store'))
    for name in ('build-config.json', 'config.json'):
        shutil.copy2(source_input / name, output / name)
    migration = Path(plan['migrated_source']['path']).resolve(strict=True)
    if digest(migration / 'proof.json') != plan['migrated_source']['proof_sha256']:
        raise ValueError('Qualified cold migration proof differs')
    for destination, replacement in (('assets/cold', 'asset'), ('bundles/cold', 'bundle')):
        shutil.rmtree(output / destination)
        shutil.copytree(migration / replacement, output / destination)
    shutil.copy2(migration / 'proof.json', output / 'source-migration-proof.json')
    source_run = json.loads((output / 'config.json').read_text())
    source_run['initialize']['manifest_sha256'] = plan['migrated_source']['asset_manifest_sha256']
    (output / 'config.json').write_text(json.dumps(source_run, indent=2) + '\n')
    (output / 'source').mkdir()
    archive = subprocess.Popen(['git', '-C', str(repository), 'archive', 'HEAD'], stdout=subprocess.PIPE)
    try:
        subprocess.run(['tar', '-x', '-C', str(output / 'source')], stdin=archive.stdout, check=True)
    finally:
        assert archive.stdout is not None
        archive.stdout.close()
    if archive.wait() != 0:
        raise RuntimeError('Source Git archive failed')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    verify_source(output, plan)
    original_build = json.loads((output / 'build-config.json').read_text())
    original_run = json.loads((output / 'config.json').read_text())
    control, treatment, run = derive_pair(original_build, original_run, plan)
    for name, build in zip(ARMS, (control, treatment), strict=True):
        directory = output / name
        directory.mkdir()
        (directory / 'build-config.json').write_text(json.dumps(build, indent=2) + '\n')
        (directory / 'config.json').write_text(json.dumps(run, indent=2) + '\n')
    proof = {'experiment': plan['experiment'], 'source_revision': subprocess.check_output(
        ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_policy_sha256': plan['source_checkpoint_sha256'],
        'control_weights': plan['control_weights'], 'treatment_weights': plan['treatment_weights'],
        'only_arm_difference': 'opponent_weights indices 9 and 12'}
    (output / 'intent.json').write_text(json.dumps(proof, indent=2) + '\n')
    for path in (output, *output.rglob('*')):
        if path.is_symlink():
            raise ValueError('Staged input cannot contain symlinks')
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | (0o555 if path.is_dir() else 0o444))
    (output / 'seal.json').write_text(json.dumps(hashes(output), indent=2) + '\n')
    return proof


def validate_probe_receipt(path: Path, inputs: Path, plan: dict,
                           expected_input_seal_sha256: str | None = None) -> dict:
    receipt = json.loads(path.read_text())
    probe = plan['required_treatment_probe']
    audit = receipt.get('audit', {})
    counts = audit.get('opponent_counts_by_seat', {})
    reward = audit.get('reward_audit', {})
    if (receipt.get('experiment') != plan['experiment']
            or receipt.get('source_policy_sha256') != plan['source_checkpoint_sha256']
            or receipt.get('input_seal_sha256') != (
                expected_input_seal_sha256 or digest(inputs / 'seal.json'))
            or receipt.get('opponent_weights') != plan['treatment_weights']
            or receipt.get('probe_seed') != probe['seed']
            or 'H100' not in receipt.get('gpu_model', '')
            or audit.get('environment_steps') != probe['steps']
            or audit.get('environment_count') != plan['training']['parallel_games']
            or audit.get('horizon') != plan['training']['horizon']
            or audit.get('minibatch') != plan['training']['minibatch']
            or audit.get('replay_ratio') != plan['training']['replay_ratio']
            or len(audit.get('epoch_uptime', [])) < 4
            or audit.get('steady_sps', 0) < probe['minimum_steady_sps']
            or audit.get('illegal_actions') != 0
            or any(reward.get(key, 1) != 0 for key in (
                'nonfinite_rewards', 'native_clipped_rewards', 'native_clipped_terminal_rewards'))
            or len(counts) != probe['required_opponents_with_both_seats']
            or any(row.get('0', 0) <= 0 or row.get('1', 0) <= 0 for row in counts.values())):
        raise ValueError('Treatment-specific H100 throughput and opponent probe is not qualified')
    return receipt


def source_gate(inputs: Path, output: Path, sampler: dict, call) -> dict:
    from integrations.launch_spatial_selfplay_training import source_sampling_gate_report
    source_match = output / 'source-sampling-match'
    call('evaluate_spatial_frozen_match', ['--bundle', inputs / 'bundles/cold',
         '--opponent-bundle', inputs / 'bundles/cold', '--games', 512, '--pool-size', 512,
         '--seed', 10441691, '--sample-seed', 10441693,
         '--sampling-temperature', sampler['move_temperature'],
         '--split-sampling-temperature', sampler['split_temperature'],
         '--early-route-temperature', sampler['early_route_temperature'],
         '--early-route-turns', sampler['early_route_turns'],
         '--neutral-route-bias', sampler['neutral_route_bias'],
         '--weak-owned-route-penalty', sampler['weak_owned_route_penalty'],
         '--doomed-attack-route-penalty', sampler['doomed_attack_route_penalty'],
         '--output', source_match], name='source-sampling', seconds=600, directory=output)
    return source_sampling_gate_report(source_match)


def verify_gate(report: dict, sampler: dict, source: str, seed: int, sample_seed: int) -> None:
    expected_sampler = dict(sampler, log_gap_scale=sampler.get('log_gap_scale', 0.0))
    if (report['gate_mode'] != 'same_sampler_source'
            or report['source_sha256'] != source or report['opponent_sha256'] != source
            or report['sampler'] != expected_sampler or report['opponent_sampler'] != expected_sampler
            or report['games'] != 512 or report['seat_counts'] != {'0': 256, '1': 256}
            or sum(report['wld']) != 512 or report['unique_initial_maps'] <= 0
            or report['match_seed'] != seed or report['sample_seed'] != sample_seed
            or report['held_out'] is not True or report['coworld_classic_rules'] is not True):
        raise ValueError('Source gate lacks complete matched sampler and Classic evidence')


def run_probe(inputs: Path, output: Path) -> None:
    """Qualify the exact treatment pool before either long matched PPO arm."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    inputs = inputs.resolve(strict=True)
    if hashes(inputs) != json.loads((inputs / 'seal.json').read_text()):
        raise ValueError('Staged probe input differs from its seal')
    plan = load_plan(inputs / 'plan.json')
    verify_source(inputs, plan)
    original_build = json.loads((inputs / 'build-config.json').read_text())
    original_run = json.loads((inputs / 'config.json').read_text())
    _, treatment, run = derive_pair(original_build, original_run, plan)
    if treatment != json.loads((inputs / 'treatment/build-config.json').read_text()):
        raise ValueError('Treatment config differs from preregistration')
    run['total_timesteps'] = plan['required_treatment_probe']['steps']
    run['seed'] = plan['required_treatment_probe']['seed']
    from integrations.cuda_runtime_binding import configure
    configure()
    from integrations.slurm_s3_job import gpu_query, verify_gpu_idle, visible_gpu_identity
    identity = visible_gpu_identity()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'gpu-idle-attempt.json').write_text(json.dumps({
        'uuid': identity['uuid'],
        'compute_pid_rows_before_jax_import': gpu_query('--id=' + identity['uuid'],
            '--query-compute-apps=pid', '--format=csv,noheader,nounits').splitlines(),
    }, indent=2) + '\n')
    gpu = verify_gpu_idle(identity)
    from integrations.classic_position_curriculum import configure_positions
    from integrations.policy_execution import training_audit
    from integrations.policy_trial import Trial
    gpu_model = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                        text=True).strip()
    if 'H100' not in gpu_model:
        raise ValueError('Treatment probe requires the preregistered H100')
    os.environ['GENERALS_ALLOCATED_GPU_UUID'] = gpu['uuid']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    probe = output / 'probe'
    probe.mkdir()
    (output / 'gpu-preflight.json').write_text(json.dumps(gpu, indent=2) + '\n')
    configure_positions(treatment['python_environment']['options'], inputs / 'curriculum/manifest.json')
    (output / 'build-config.json').write_text(json.dumps(treatment, indent=2) + '\n')
    config = probe / 'config.json'
    config.write_text(json.dumps(run, indent=2) + '\n')
    trial = Trial(inputs, output)
    trial.build()
    os.environ['PYTHONFAULTHANDLER'] = '1'
    os.environ['METTA_GATE_TRACE'] = '1'
    trial.sampling_gate('control')
    sampler = trial.sampler
    proven_gate = output / 'control/sampling-gate.json'
    verify_gate(json.loads(proven_gate.read_text()), sampler, plan['source_checkpoint_sha256'], 51231, 17441)
    shutil.copy2(proven_gate, probe / 'sampling-gate.json')

    def stress_call(module, args, *, name, seconds, directory):
        trial.call(module, args, name=name, seconds=seconds, arm='probe')

    stress_gate = source_gate(inputs, output, sampler, stress_call)
    verify_gate(stress_gate, sampler, plan['source_checkpoint_sha256'], 10441691, 10441693)
    (probe / 'hard-seed-sampling-gate.json').write_text(json.dumps(stress_gate, indent=2) + '\n')
    # Native preflight verifies the actual source model fingerprint.
    trial.call('launch_spatial_selfplay_training', ['preflight', '--build', output / 'build',
         '--config', config, '--output', probe / 'run'],
         name='preflight', seconds=300, arm='probe')
    trial.call('launch_spatial_selfplay_training', ['train', '--build', output / 'build',
         '--config', config, '--output', probe / 'run'], name='train', seconds=900,
         arm='probe', training=True)
    audit = training_audit(probe, run)
    if audit['steady_sps'] < plan['required_treatment_probe']['minimum_steady_sps']:
        raise ValueError('Treatment throughput probe is below 30,000 steady SPS')
    (output / 'QUALIFIED.json').write_text(json.dumps({
        'experiment': plan['experiment'], 'source_policy_sha256': plan['source_checkpoint_sha256'],
        'input_seal_sha256': digest(inputs / 'seal.json'),
        'opponent_weights': plan['treatment_weights'], 'probe_seed': run['seed'],
        'gpu_model': gpu_model,
        'proven_source_gate_sha256': digest(proven_gate),
        'hard_seed_source_gate_sha256': digest(probe / 'hard-seed-sampling-gate.json'),
        'gpu': gpu, 'audit': audit,
    }, indent=2) + '\n')


def run_pair(inputs: Path, output: Path, qualified_probe: Path,
             qualified_probe_sha256: str, qualified_probe_job_id: str) -> None:
    """Train both arms on one GPU and evaluate on common fresh Classic maps."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    inputs = inputs.resolve(strict=True)
    if hashes(inputs) != json.loads((inputs / 'seal.json').read_text()):
        raise ValueError('Staged input differs from its seal')
    plan = load_plan(inputs / 'plan.json')
    verify_source(inputs, plan)
    original_build = json.loads((inputs / 'build-config.json').read_text())
    original_run = json.loads((inputs / 'config.json').read_text())
    control, treatment, run = derive_pair(original_build, original_run, plan)
    for name, expected in zip(ARMS, (control, treatment), strict=True):
        if json.loads((inputs / name / 'build-config.json').read_text()) != expected:
            raise ValueError('Staged arm config differs from frozen comparison')
        if json.loads((inputs / name / 'config.json').read_text()) != run:
            raise ValueError('Staged PPO config differs from matched control')
    if (qualified_probe_job_id != QUALIFIED_PROBE_JOB_ID
            or QUALIFIED_PROBE_SHA256 is None
            or qualified_probe_sha256 != QUALIFIED_PROBE_SHA256
            or digest(qualified_probe) != qualified_probe_sha256):
        raise ValueError('Long run requires the exact job-izcgj treatment probe marker')
    probe_receipt = validate_probe_receipt(
        qualified_probe, inputs, plan, QUALIFIED_PROBE_INPUT_SEAL_SHA256)
    from integrations.cuda_runtime_binding import configure
    configure()
    from integrations.slurm_s3_job import gpu_query, verify_gpu_idle, visible_gpu_identity
    identity = visible_gpu_identity()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'gpu-idle-attempt.json').write_text(json.dumps({
        'uuid': identity['uuid'],
        'compute_pid_rows_before_jax_import': gpu_query('--id=' + identity['uuid'],
            '--query-compute-apps=pid', '--format=csv,noheader,nounits').splitlines(),
    }, indent=2) + '\n')
    gpu = verify_gpu_idle(identity)
    from integrations.classic_position_curriculum import configure_positions
    from integrations.policy_execution import training_audit
    from integrations.policy_trial import Trial
    os.environ['GENERALS_ALLOCATED_GPU_UUID'] = gpu['uuid']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    (output / 'gpu-preflight.json').write_text(json.dumps(gpu, indent=2) + '\n')
    source = inputs / 'source'

    for name in ARMS:
        staged = inputs / name
        target = output / name
        target.mkdir()
        probe = target / 'probe'
        probe.mkdir()
        build = json.loads((staged / 'build-config.json').read_text())
        configure_positions(build['python_environment']['options'], inputs / 'curriculum/manifest.json')
        (target / 'build-config.json').write_text(json.dumps(build, indent=2) + '\n')
        config = probe / 'config.json'
        shutil.copy2(staged / 'config.json', config)
        trial = Trial(inputs, target)
        trial.build()
        trial.sampling_gate('control')
        gate = target / 'control/sampling-gate.json'
        verify_gate(json.loads(gate.read_text()), trial.sampler,
                    plan['source_checkpoint_sha256'], 51231, 17441)
        shutil.copy2(gate, probe / 'sampling-gate.json')
        trial.call('launch_spatial_selfplay_training', ['preflight', '--build', target / 'build',
             '--config', config, '--output', probe / 'run'], name='preflight', seconds=300,
             arm='probe')
        trial.call('launch_spatial_selfplay_training', ['train', '--build', target / 'build',
             '--config', config, '--output', probe / 'run'], name='train', seconds=1800,
             arm='probe', training=True)
        training_audit(probe, json.loads(config.read_text()))
        checkpoint = probe / f"run/checkpoints/metta_generals/run/{plan['training']['steps_per_arm']:016d}.bin"
        sampler_path = target / 'sampler.json'
        sampler_path.write_text(json.dumps(trial.sampler, indent=2) + '\n')
        trial.call('publish_policy_asset', ['--build', target / 'build/build.json', '--training',
             probe / 'run/training.json', '--checkpoint', checkpoint, '--sha256', digest(checkpoint),
             '--sampler', sampler_path, '--factory-source', source / 'integrations/generals_fabric.py',
             '--output', target / 'asset'], name='publish', seconds=600, arm='probe')
        asset = target / 'asset/asset.json'
        trial.call('export_spatial_policy_bundle', ['--asset', asset, '--manifest-sha256', digest(asset),
             '--factory-source', source / 'integrations/generals_fabric.py', '--output', target / 'bundle'],
             name='export', seconds=600, arm='probe')
    dev = plan['development']
    trial = Trial(inputs, output)
    for name in ARMS:
        trial.call('evaluate_spatial_population', ['--bundle', output / name / 'bundle',
             '--population-build', output / 'control/build/build.json', '--games', dev['games_per_arm'],
             '--pool-size', dev['pool_size'], '--seed', dev['map_seed'],
             '--sample-seed', dev['sample_seed'], '--output', output / ('eval-' + name)],
             name='evaluate-' + name, seconds=1800)
    comparison = output / 'comparison.json'
    trial.call('analyze_spatial_population_pair', ['--baseline', output / 'eval-control',
         '--candidate', output / 'eval-treatment', '--seed', dev['bootstrap_seed'],
         '--bootstrap-resamples', 10000, '--output', comparison],
         name='compare', seconds=90)
    report = json.loads(comparison.read_text())
    strata = report['by_opponent_and_seat']
    hard_names = (plan['opponent_order'][9], plan['opponent_order'][12])
    hard_rows = [strata[name][str(side)] for name in hard_names for side in (0, 1)]
    hard_gain = sum(row['games'] * row['paired_signed_score_delta'] for row in hard_rows) / sum(
        row['games'] for row in hard_rows)
    broad = [row for seats in strata.values() for row in seats.values() if row['games'] >= 100]
    selected = (report['initial_state_cluster_ci95'][0] > 0 and hard_gain > 0
                and all(row['paired_signed_score_delta'] >= -0.10 for row in broad))
    (output / 'COMPLETED.json').write_text(json.dumps({'experiment': plan['experiment'],
        'development_only': True, 'hard_opponent_paired_signed_score_delta': hard_gain,
        'treatment_selected_for_independent_confirmation': selected,
        'comparison_sha256': digest(comparison), 'qualified_probe_sha256': digest(qualified_probe),
        'qualified_probe_job_id': qualified_probe_job_id,
        'qualified_probe_steady_sps': probe_receipt['audit']['steady_sps'],
        'gpu': gpu}, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('prepare')
    stage.add_argument('--source-input', type=Path, required=True)
    stage.add_argument('--repository', type=Path, required=True)
    stage.add_argument('--output', type=Path, required=True)
    probe = commands.add_parser('probe')
    probe.add_argument('--input', type=Path, required=True)
    probe.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--input', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--qualified-probe', type=Path, required=True)
    run.add_argument('--qualified-probe-sha256', required=True)
    run.add_argument('--qualified-probe-job-id', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.source_input, args.repository, args.output), indent=2))
    elif args.command == 'probe':
        run_probe(args.input, args.output)
    else:
        run_pair(args.input, args.output, args.qualified_probe,
                 args.qualified_probe_sha256, args.qualified_probe_job_id)


if __name__ == '__main__':
    main()

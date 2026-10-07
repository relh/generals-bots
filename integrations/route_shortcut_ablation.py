"""One frozen source-versus-route-shortcut ablation development panel."""
import hashlib
import json
import os
from pathlib import Path
import resource
import struct


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = Path('/work/input')
    output = Path('/output/generals')
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['schema'] == 'generals-route-shortcut-ablation-v1'
    assert plan['games_per_arm'] == 4096
    assert [plan[k] for k in ('map_seed', 'sample_seed', 'bootstrap_seed')] == [11007101, 11007103, 11007111]
    for relative, digest in json.loads((root / 'seal.json').read_text()).items():
        if sha(root / relative) != digest:
            raise ValueError('Sealed input changed: ' + relative)
    for arm in ('source', 'candidate'):
        bundle = root / 'bundles' / arm
        metadata = json.loads((bundle / 'asset.json').read_text())
        assert metadata['policy_sha256'] == plan[arm + '_checkpoint_sha256']
        assert sha(bundle / 'policy.bin') == metadata['policy_sha256']
        assert metadata['sampler'] == plan['sampler']
    assert plan['source_checkpoint_sha256'] == 'f4ef5616f76131bb23eee42c25b450353de73832e609ec63499887c7a2634d14'
    assert sha(root / 'bundles/source/spatial-policy.json') == plan['source_bundle_manifest_sha256']
    assert sha(root / 'population-build.json') == plan['original_population_build_sha256']
    assert sha(root / 'source/generals/core/coworld_game.py') == plan['official_engine_sha256']
    for index, expected in plan['frozen_bundle_manifest_sha256'].items():
        assert sha(root / 'bundles/frozen' / index / 'spatial-policy.json') == expected
    before = (root / 'bundles/source/policy.bin').read_bytes()
    after = (root / 'bundles/candidate/policy.bin').read_bytes()
    assert len(before) == len(after) and len(before) % 4 == 0
    changed_indices = [index for index, (old, new) in enumerate(zip(
        struct.iter_unpack('<I', before), struct.iter_unpack('<I', after))) if old != new]
    assert len(changed_indices) == plan['candidate_native_changed_words'] == 8
    assert changed_indices == sorted(plan['candidate_native_changed_indices'])
    operation = json.loads((root / 'evidence/operation.json').read_text())
    assert changed_indices == sorted(operation['changed_native_scalar_indices'])
    floats_before = [x[0] for x in struct.iter_unpack('<f', before)]
    floats_after = [x[0] for x in struct.iter_unpack('<f', after)]
    for row in operation['directions']:
        full, half = row['full_prior_index'], row['half_prior_index']
        of = floats_before[row['full_output_weight_index']]
        oh = floats_before[row['half_output_weight_index']]
        expected = struct.unpack('<f', struct.pack('<f', floats_before[half] - floats_before[full] * of / oh))[0]
        assert floats_after[full] == 0 and floats_after[half] == expected
    from integrations.cuda_runtime_binding import configure
    from integrations.slurm_s3_job import visible_gpu_identity, verify_gpu_idle
    from integrations.policy_execution import execute
    configure()
    gpu = verify_gpu_idle(visible_gpu_identity())
    os.environ['GENERALS_ALLOCATED_GPU_UUID'] = gpu['uuid']
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
    for name in ('tmp', 'cache', 'jax-cache', 'fabric-verify'):
        (Path('/work') / name).mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'gpu.json').write_text(json.dumps(gpu, indent=2) + '\n')
    def run(module, args, name, seconds):
        print('STAGE ' + name, flush=True)
        execute(module, args, source=root / 'source', output=output,
                sampler=plan['sampler'], name=name, seconds=seconds)
    for arm in ('source', 'candidate'):
        run('evaluate_spatial_population', [
            '--bundle', root / 'bundles' / arm,
            '--population-build', root / 'population-build.json',
            '--games', 4096, '--pool-size', 4096, '--seed', plan['map_seed'],
            '--sample-seed', plan['sample_seed'], '--destination-audit',
            '--output', output / arm], 'evaluate-' + arm, 1200)
    comparison = output / 'comparison.json'
    run('analyze_spatial_population_pair', [
        '--baseline', output / 'source', '--candidate', output / 'candidate',
        '--output', comparison, '--seed', plan['bootstrap_seed'],
        '--bootstrap-resamples', 10000], 'compare', 180)
    result = json.loads(comparison.read_text())
    broad_failures = [dict(opponent=name, seat=seat, **row)
                      for name, seats in result['by_opponent_and_seat'].items()
                      for seat, row in seats.items()
                      if row['games'] >= 100 and row['paired_signed_score_delta'] < -0.10]
    evaluations = {}
    for arm in ('source', 'candidate'):
        path = output / arm / 'evaluation.json'
        record = json.loads(path.read_text())
        assert record['games'] == 4096 and record['coworld_classic_rules']
        assert record['checkpoint_sha256'] == plan[arm + '_checkpoint_sha256']
        assert len(record['by_opponent_and_seat']) == 13
        for seats in record['by_opponent_and_seat'].values():
            assert seats['0']['games'] == seats['1']['games'] > 0
        evaluations[arm] = {'sha256': sha(path), **{k: record[k] for k in ('wins', 'losses', 'draws')}}
    selected = result['initial_state_cluster_ci95'][0] > 0 and not broad_failures
    report = {'schema': 'generals-route-shortcut-development-v1',
              'plan_sha256': sha(root / 'plan.json'), 'comparison_sha256': sha(comparison),
              'evaluation': evaluations, 'selected_for_independent_confirmation': selected,
              'broad_stratum_failures': broad_failures,
              'paired_signed_score_delta': result['paired_signed_score_delta'],
              'initial_state_cluster_ci95': result['initial_state_cluster_ci95'],
              'scope': 'Development evidence only; positive result requires independent confirmation before hosted qualification.'}
    (output / 'COMPLETED.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

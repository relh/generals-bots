"""Prepare confirmation input only from exact successful positive development output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(collection, base_plan, output):
    terminal = json.loads((collection / 'terminal.json').read_text())
    metadata = json.loads((collection / 'artifact.json').read_text())
    if terminal['job_id'] != 'job-wwvk3' or terminal['status'] != 'succeeded':
        raise ValueError('Exact development job must succeed before confirmation staging')
    if terminal['artifact']['artifact_id'] != metadata['artifact_id'] or metadata['state'] != 'ready':
        raise ValueError('Development provider artifact identity differs')
    archive = collection / 'output.tar'
    if sha(archive) != metadata['sha256'] or archive.stat().st_size != metadata['size_bytes']:
        raise ValueError('Development archive differs from provider checksum')
    with tarfile.open(archive) as tar:
        members = {m.name.removeprefix('./'): m for m in tar.getmembers() if m.isfile()}
        def read(name):
            matches = [m for key, m in members.items() if key == name or key.endswith('/' + name)]
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous development artifact: ' + name)
            return tar.extractfile(matches[0]).read()
        completed_raw = read('COMPLETED.json')
        completed = json.loads(completed_raw)
        comparison_raw = read('comparison.json')
        comparison = json.loads(comparison_raw)
        if completed['comparison_sha256'] != hashlib.sha256(comparison_raw).hexdigest():
            raise ValueError('Development comparison checksum differs')
        if not completed['selected_for_independent_confirmation']:
            raise ValueError('Rejected development result cannot stage confirmation')
        if comparison['games'] != 4096 or comparison['initial_state_cluster_ci95'][0] <= 0:
            raise ValueError('Development improvement gate did not pass')
        if any(row['games'] >= 100 and row['paired_signed_score_delta'] < -0.10
               for seats in comparison['by_opponent_and_seat'].values() for row in seats.values()):
            raise ValueError('Development broad-stratum guard did not pass')
        plan = json.loads(base_plan.read_text())
        if sha(base_plan) != completed['plan_sha256']:
            raise ValueError('Frozen development plan differs')
        if plan['source_revision'] != '601177976538bc8cd31ceea014496a35662e1f8d':
            raise ValueError('Unexpected development source')
        for arm in ('source', 'candidate'):
            raw = read(arm + '/evaluation.json')
            record = json.loads(raw)
            if hashlib.sha256(raw).hexdigest() != completed['evaluation'][arm]['sha256']:
                raise ValueError('Development evaluation checksum differs')
            if record['checkpoint_sha256'] != plan[arm + '_checkpoint_sha256'] or record['games'] != 4096:
                raise ValueError('Development arm identity differs')
    plan.update(schema='generals-route-shortcut-confirmation-v1',
                scope='Independent confirmation of frozen alpha0 candidate; no training, tuning, or promotion.',
                source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                map_seed=11008101, sample_seed=11008103, bootstrap_seed=11008111,
                development_job='job-wwvk3', development_artifact_sha256=metadata['sha256'],
                development_completed_sha256=hashlib.sha256(completed_raw).hexdigest())
    output.mkdir(parents=True, exist_ok=False)
    proof = output / 'development-completed.json'
    proof.write_bytes(completed_raw)
    plan['evidence_files']['development-completed.json'] = {'path': str(proof.resolve()), 'sha256': sha(proof)}
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-collection', type=Path, required=True)
    parser.add_argument('--development-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.development_collection, args.development_plan, args.output)


if __name__ == '__main__':
    main()

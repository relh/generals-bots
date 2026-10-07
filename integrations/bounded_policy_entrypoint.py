"""Bound native experiment process groups and retain complete or partial evidence."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tarfile
import time


def collect(results, output, complete, elapsed, kind, completion):
    retained = []
    if results.exists():
        for path in sorted(results.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(results)
            if 'build' in rel.parts and 'source' in rel.parts:
                tail = Path(*rel.parts[rel.parts.index('source') + 1:])
                if not (tail.parts[0] == 'config' or tail.parts[0] == 'src' and path.suffix in ('.cu', '.cuh', '.h')):
                    continue
            retained.append((path, rel))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(schema=f'generals-{kind}-result-v1', complete=complete,
                   elapsed_seconds=elapsed, files={str(rel): sha(path) for path, rel in retained})
    (output / 'collection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    archive_path = output / 'results.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path, rel in retained:
            archive.add(path, arcname='generals/' + str(rel), recursive=False)
    if complete:
        (output / 'COMPLETED.json').write_text(json.dumps(dict(
            schema=f'generals-{kind}-completed-v1', results_sha256=sha(archive_path),
            plan_sha256=sha(results / 'plan.json'), **completion(results))) + '\n')


def run(commands, results, output, seconds, *, kind, completion):
    """Bound the coordinator; it cleans each native child group on interruption."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output.mkdir(parents=True, exist_ok=True)
    results.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    complete = False
    child = None
    def stop(signum, frame):
        raise InterruptedError(f'Continuation interrupted by signal {signum}')
    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for command, env in commands:
            remaining = seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Experiment execution budget exhausted')
            child = subprocess.Popen(command, env=env, start_new_session=True)
            code = child.wait(timeout=remaining)
            if code:
                raise RuntimeError(f'Experiment failed, exit {code}')
            child = None
        completion(results)  # Validate all success markers before marking collection complete.
        complete = True
    except BaseException as exc:
        (output / 'FAILED.json').write_text(json.dumps(dict(error=str(exc), elapsed=time.monotonic()-start)) + '\n')
        raise
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
        for sig, handler in old.items():
            signal.signal(sig, handler)
        collect(results, output, complete, time.monotonic()-start, kind, completion)


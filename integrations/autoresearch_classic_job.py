"""One finite H100 qualification, with owned-child shutdown before result capture.

The immutable image supplies /opt/pilot/input.tar.gz, input-package.json and
the CPU-qualified SM90 build. Autoresearch captures GMN_OUTPUT_DIR on exit.
This entry point never creates nodes, submits jobs, changes billing or retries.
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from integrations.slurm_s3_job import check_space, extract_input


@contextmanager
def blocked_termination():
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGUSR1})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def run_phase(argv, log, seconds, env):
    """Every descendant shares our child's new group, not the wrapper's group."""
    with Path(log).open('xb') as output:
        signals = {signal.SIGTERM, signal.SIGINT, signal.SIGUSR1}
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        try:
            process = subprocess.Popen(
                argv, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous),
            )
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            raise
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            code = process.wait(timeout=seconds)
            if code:
                raise RuntimeError(f'Qualification phase failed with exit {code}; inspect its retained log')
        finally:
            with blocked_termination():
                # Even a terminated leader can leave descendants. Signal only the
                # process group we created; never shared/user-wide process matches.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                # A leader may exit before a resistant descendant; kill any tail.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Linux container descendants may be reparented to PID1. They are
                # not waitpid children; confirm no non-zombie group member remains.
                if Path('/proc').is_dir():
                    deadline = time.monotonic() + 10
                    while True:
                        alive = False
                        for entry in Path('/proc').iterdir():
                            if not entry.name.isdigit():
                                continue
                            try:
                                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                            except (FileNotFoundError, ProcessLookupError, PermissionError):
                                continue
                            if int(fields[2]) == process.pid and fields[0] != 'Z':
                                alive = True
                                break
                        if not alive:
                            break
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Owned process group did not stop; preserve all output')
                        time.sleep(.02)



def main():
    output_base = Path(os.environ['GMN_OUTPUT_DIR'])
    if not output_base.is_absolute() or not output_base.is_dir():
        raise ValueError('Expected the platform-owned output mount')
    output = output_base / 'generals'
    output.mkdir(exist_ok=False)
    work = Path('/work')
    work.mkdir(exist_ok=False)
    (work / 'out').symlink_to(output, target_is_directory=True)
    (work / 'tmp').mkdir()
    start = time.monotonic()
    receipt = dict(state='preparing', phases=[], gpu_count=1, additional_steps=8388608)

    def stopped(signum, _frame):
        raise RuntimeError(f'Qualification interrupted by signal {signum}')

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stopped)
    try:
        package = json.loads(Path('/opt/pilot/input-package.json').read_text())
        archive = Path('/opt/pilot/input.tar.gz')
        if hashlib.sha256(archive.read_bytes()).hexdigest() != package['archive_sha256']:
            raise ValueError('Frozen input archive differs')
        check_space(work, 32 * 1024**3, 60000)
        inputs = work / 'input'
        inputs.mkdir()
        extract_input(archive, inputs, package['input_unpacked_bytes'], package['input_members'])
        Path('/recovery').symlink_to(inputs / 'recovery', target_is_directory=True)
        env = dict(os.environ, GENERALS_COMPUTE_PROVIDER='autoresearch',
                   GENERALS_PILOT_PARALLEL_GAMES='2048', GENERALS_PILOT_STEPS='8388608',
                   GENERALS_FULL_ACTION_TEMPERATURE='1', GENERALS_LOG_GAP_SCALE='4',
                   GENERALS_EVAL_SEED='40913', GENERALS_EVAL_SAMPLE_SEED='10211',
                   GENERALS_PILOT_CONTINUATION_MANIFEST='/work/input/continuation/manifest.json',
                   GENERALS_PILOT_POSITION_MANIFEST='/work/input/curriculum/manifest.json',
                   PYTHONPATH='/work/input/source:/opt/generals-source', TMPDIR='/work/tmp',
                   XDG_CACHE_HOME='/work/tmp/cache', NVCC_ARCH='sm_90')
        env.pop('JAX_PLATFORMS', None)
        for name, budget in [('smoke', 180), ('sampling_gate', 300), ('train', 840), ('evaluate', 1200)]:
            run_phase([sys.executable, '-m', 'integrations.portable_classic_pilot', name],
                      output / f'{name}-wrapper.log', budget, env)
            receipt['phases'].append(name)
            if name == 'smoke':
                fixture = Path('/opt/pilot/build')
                build = json.loads((fixture / 'build.json').read_text())
                if build['config'] != json.loads((output / 'build-config.json').read_text()):
                    raise ValueError('CPU-qualified native build configuration differs')
                if hashlib.sha256((fixture / 'puffer').read_bytes()).hexdigest() != build['binary_sha256']:
                    raise ValueError('CPU-qualified native binary differs')
                shutil.copytree(fixture, output / 'build')
            (output / 'progress.json').write_text(json.dumps(receipt) + '\n')
        receipt['state'] = 'completed'
    except BaseException as error:
        receipt.update(state='failed', error_type=type(error).__name__)
        raise
    finally:
        # run_phase has already stopped/waited its owned process before here.
        receipt['elapsed_seconds'] = time.monotonic() - start
        (output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()

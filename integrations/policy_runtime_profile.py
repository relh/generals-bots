"""Bounded execution profiling; never selects policies or evaluates outcomes."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def cpu_snapshot():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    files = {}
    for name in ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective'):
        path = Path('/sys/fs/cgroup') / name
        files[name] = path.read_text().strip() if path.exists() else None
    capacity = float(len(affinity))
    if files['cpu.max']:
        quota, period = files['cpu.max'].split()
        if quota != 'max':
            capacity = min(capacity, int(quota) / int(period))
    cpuinfo = Path('/proc/cpuinfo')
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                  if line.startswith('model name')), None) if cpuinfo.exists() else None
    return dict(affinity=affinity, effective_cpus=capacity, cgroup=files, model=model,
                thread_environment={k:v for k,v in os.environ.items() if k.startswith(('OMP_', 'OPENBLAS_', 'MKL_'))})


def telemetry(output, uuid):
    fields = ('timestamp,uuid,memory.used,utilization.gpu,name,driver_version,pstate,'
              'clocks.current.sm,clocks.current.memory,power.draw,power.limit,'
              'temperature.gpu,clocks_event_reasons.active')
    result = subprocess.run(['nvidia-smi', '--id='+uuid, '--query-gpu='+fields, '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=10, check=True)
    values = next(csv.reader([result.stdout.strip()], skipinitialspace=True))
    if len(values) != len(fields.split(',')):
        raise ValueError('GPU telemetry field count differs')
    with (output/'gpu.csv').open('a') as f:
        f.write(', '.join(values[:4])+'\n')
    with (output/'hardware.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_unix=time.time(), cpu=cpu_snapshot(),
                               gpu=dict(zip(fields.split(','),values))))+'\n')


def replay_fixture(root):
    """Fixed public views from existing sealed replays; no new games or seeds."""
    import numpy as np
    from integrations.softmax.engine import Match
    manifest = json.loads((root/'leader-replay-manifest.json').read_text())
    grids, turns, labels = [], [], []
    for index, item in enumerate(manifest[:4]):
        blob = (root/'leader-replays'/(item['episode_id']+'.bin')).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            raise ValueError('Profiling replay checksum differs')
        replay = json.loads(gzip.decompress(blob))
        if replay['ruleset'] != 'classic':
            raise ValueError('Profiling requires Classic public views')
        match = Match(replay['seed'])
        for turn, receipt in enumerate(replay['turns'][:201]):
            frame = match.frame()
            if any(frame[k] != replay['frames'][turn][k] for k in
                   ('turn','type_grid','owner_grid','army_grid','army','land')):
                raise ValueError('Profiling replay frame differs from pinned engine')
            if turn in (0,25,99,100,150,200):
                for side in (0,1):
                    obs = match.observation(side)
                    grid = np.zeros((3,21,21), np.int32); grid[0] = 2
                    for plane,key in enumerate(('type_grid','owner_grid','army_grid')):
                        grid[plane,:obs['height'],:obs['width']] = obs[key]
                    grids.append(grid); turns.append(turn); labels.append((index,turn,side))
            if not receipt['applied'] or receipt['turn'] != match.turn:
                raise ValueError('Profiling replay receipt mismatch')
            match.advance(receipt['actions'])
    if not grids:
        raise ValueError('No public profiling views')
    rows = np.arange(906) % len(grids)  # Exact 4096-environment training siege population.
    return np.full((906,2),21,np.int32), np.asarray(turns,np.int32)[rows], np.asarray(grids,np.int32)[rows], labels


def select_siege_workers(replay_root, output):
    import numpy as np
    from integrations.classic_siege_native import ClassicSiegeBatch, compile_library
    output.mkdir(parents=True, exist_ok=True)
    cpu = cpu_snapshot()
    (output/'cpu-before.json').write_text(json.dumps(cpu,indent=2)+'\n')
    candidates = [n for n in (1,2,4,8) if n <= cpu['effective_cpus']]
    if not candidates:
        candidates = [1]  # A fractional CPU quota still needs one calling thread.
    library = compile_library(output/'opponent.so')
    dims, turns, grids, labels = replay_fixture(replay_root)
    serial = ClassicSiegeBatch(library)
    memory = serial(dims,turns,grids,serial.initial_memory(len(turns)))[1]
    inputs = (dims,turns,grids,memory)
    hashes = {name:hashlib.sha256(x.tobytes()).hexdigest() for name,x in zip(('dimensions','turns','grids','memory'),inputs)}
    np.savez(output/'public-fixture.npz',dimensions=dims,turns=turns,grids=grids,memory=memory)
    reference = serial(*inputs)
    actors = {n:ClassicSiegeBatch(library,workers=n) for n in candidates}
    samples = {n:[] for n in candidates}
    def timeout(signum, frame):
        raise TimeoutError('Native siege worker profiling exceeded 30 seconds')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL,30)
    try:
        for actor in actors.values():
            actual = actor(*inputs)
            for a,b in zip(actual,reference):
                np.testing.assert_array_equal(a,b)
        for repeat in range(5):
            for n in candidates[repeat % len(candidates):]+candidates[:repeat % len(candidates)]:
                start = time.perf_counter()
                for _ in range(10):
                    actors[n](*inputs)
                samples[n].append((time.perf_counter()-start)/10)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
    for name,x in zip(hashes,inputs):
        if hashlib.sha256(x.tobytes()).hexdigest() != hashes[name]:
            raise ValueError('Native worker profile mutated fixed public inputs')
    medians = {n:statistics.median(values) for n,values in samples.items()}
    best = min(medians.values())
    tied = [n for n in candidates if medians[n] <= best*1.02]
    chosen = min(tied,key=lambda n:(n != 4,n))
    report = dict(selected_workers=chosen,cpu=cpu,cpu_after=cpu_snapshot(), candidates=candidates,
                  samples_seconds=samples,medians_seconds=medians,tie_rule='within 2% of fastest, prefer 4 then smaller',
                  input_sha256=hashes,labels=labels,rows=len(turns),action_and_memory_equal=True,
                  scope='CPU callback only, fixed public replay inputs; not training SPS or policy strength')
    (output/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    return chosen

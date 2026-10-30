"""Bounded execution profiling; never selects policies or evaluates outcomes."""
import csv
import json
import os
from pathlib import Path
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


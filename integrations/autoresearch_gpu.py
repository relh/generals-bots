"""Single-device identity checks for the explicitly selected funded backend."""

import os
import re
from pathlib import Path

from integrations.slurm_s3_job import gpu_query


def identity(environ=None):
    env = os.environ if environ is None else environ
    if env.get('GENERALS_COMPUTE_PROVIDER') != 'autoresearch':
        raise RuntimeError('Autoresearch GPU checks require explicit provider selection')
    output = env.get('GMN_OUTPUT_DIR', '')
    cpus = env.get('GMN_CPU_LIMIT', '')
    if not output or not Path(output).is_absolute() or not cpus.isdigit() or int(cpus) <= 0:
        raise RuntimeError('Expected Autoresearch job output and CPU allocation environment')
    rows = gpu_query('--query-gpu=index,uuid,name', '--format=csv,noheader,nounits').splitlines()
    if len(rows) != 1:
        raise RuntimeError('Expected exactly one visible H100')
    fields = [v.strip() for v in rows[0].split(',')]
    if (len(fields) != 3 or not fields[0].isdigit()
            or not re.fullmatch(r'GPU-[a-fA-F0-9-]+', fields[1]) or 'H100' not in fields[2]):
        raise RuntimeError('Unexpected funded GPU identity or architecture')
    return dict(provider='autoresearch', visible_index=fields[0], uuid=fields[1], model=fields[2])


def verify_idle():
    result = identity()
    uuid = result['uuid']
    if gpu_query('--id=' + uuid, '--query-compute-apps=pid', '--format=csv,noheader,nounits'):
        raise RuntimeError('Allocated GPU already has compute processes; leaving them untouched')
    fields = [x.strip() for x in gpu_query('--id=' + uuid,
        '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits').split(',')]
    if (len(fields) != 3 or fields[0] != uuid or float(fields[1]) >= 2048 or float(fields[2]) >= 20):
        raise RuntimeError('Allocated GPU is not idle before workload startup')
    return dict(result, memory_mib=float(fields[1]), utilization_percent=float(fields[2]))

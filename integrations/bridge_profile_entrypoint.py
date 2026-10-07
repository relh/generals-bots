"""Finite diagnostic process with complete owned-result retention on failure."""
import hashlib,json,os,resource,signal,subprocess,sys,tarfile,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
from integrations.bridge_execution_probe import PLAN
OUTPUT=Path(os.environ.get('GMN_OUTPUT_DIR','/output'));OUTPUT.mkdir(parents=True,exist_ok=True)
RESULTS=Path('/work/results/bridge');RESULTS.parent.mkdir(parents=True,exist_ok=True)
started=time.monotonic();complete=False

def interrupted(signum,frame):raise InterruptedError(f'Probe interrupted by signal {signum}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
try:
    process=subprocess.Popen([sys.executable,'-u','-m','integrations.bridge_execution_probe','run',
            '--inputs','/work/input','--output',str(RESULTS)],start_new_session=True)
    try:
        code=process.wait(timeout=60*PLAN['runtime_bound']['execution_minutes'])
        if code:raise RuntimeError(f'Probe exited {code}')
        complete=True
    finally:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
except BaseException as exc:
    (OUTPUT/'FAILED.json').write_text(json.dumps(dict(error=str(exc),elapsed_seconds=time.monotonic()-started))+'\n')
    raise
finally:
    retained=[]
    for p in sorted(RESULTS.rglob('*')):
        if not p.is_file() or p.is_symlink():continue
        rel=p.relative_to(RESULTS)
        if '.git' in rel.parts:continue
        if 'build' in rel.parts and 'source' in rel.parts:
            tail=Path(*rel.parts[rel.parts.index('source')+1:])
            if not (tail.parts[0]=='config' or tail.parts[0]=='src' and p.suffix in ('.cu','.cuh','.h')):continue
        retained.append((p,rel))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=dict(complete=complete,diagnostic_only=True,elapsed_seconds=time.monotonic()-started,
                 files={str(rel):sha(p) for p,rel in retained})
    (OUTPUT/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with tarfile.open(OUTPUT/'results.tar.gz','w:gz') as archive:
        for p,rel in retained:archive.add(p,arcname='bridge/'+str(rel),recursive=False)
    if complete:
        (OUTPUT/'COMPLETED.json').write_text(json.dumps(dict(diagnostic_only=True,qualified=False,
             results_sha256=sha(OUTPUT/'results.tar.gz'),report_sha256=sha(RESULTS/'report.json'),
             plan_sha256=sha(RESULTS/'plan.json')))+'\n')

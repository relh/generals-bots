"""Finite one-shot experiment with retained failure artifacts; no submissions."""
import hashlib,json,os,signal,subprocess,sys,tarfile,time,resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from pathlib import Path

INPUT=Path('/work/input');RESULTS=Path('/work/results/generals');OUTPUT=Path(os.environ.get('GMN_OUTPUT_DIR','/output'))
OUTPUT.mkdir(parents=True,exist_ok=True);RESULTS.parent.mkdir(parents=True,exist_ok=True)
from integrations.row_rotation_trial import PLAN
EXECUTION_SECONDS = 60 * PLAN["runtime_bound"]["execution_minutes"]
if EXECUTION_SECONDS >= 60 * PLAN["runtime_bound"]["provider_minutes"]:
    raise ValueError("Execution limit must reserve time for artifact collection")
START=time.monotonic();child=None;complete=False

def stop(signum,frame):
    raise InterruptedError(f'Experiment interrupted by signal {signum}')

signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
    for phase in ('prepare','smoke','build','qualify','continue_training','evaluate'):
        remaining=EXECUTION_SECONDS-(time.monotonic()-START)
        if remaining<=0:raise TimeoutError('Experiment reached its preregistered execution limit')
        print('EXPERIMENT_PHASE '+json.dumps(dict(phase=phase,event='start')),flush=True)
        env=dict(os.environ)
        if phase=='prepare':env['JAX_PLATFORMS']='cpu'
        child=subprocess.Popen([sys.executable,'-u','-m','integrations.row_rotation_trial',phase,
              '--input',str(INPUT),'--output',str(RESULTS)],env=env,start_new_session=True)
        try:
            code=child.wait(timeout=remaining)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:child=None
        if code:raise RuntimeError(f'Experiment phase {phase} failed, exit {code}')
        print('EXPERIMENT_PHASE '+json.dumps(dict(phase=phase,event='complete',elapsed=time.monotonic()-START)),flush=True)
    complete=True
except BaseException as exc:
    (OUTPUT/'FAILED.json').write_text(json.dumps(dict(error=str(exc),elapsed=time.monotonic()-START))+'\n')
    raise
finally:
    retained=[]
    if RESULTS.exists():
        for p in sorted(RESULTS.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(RESULTS)
            # Source is reconstructible from sealed input plus pinned Puffer/framework.
            # Preserve actual generated optimizer source/config and native binary.
            parts=rel.parts
            if 'build' in parts and 'source' in parts:
                tail=Path(*parts[parts.index('source')+1:]).as_posix()
                if tail not in ('src/algo.cu','config/default.ini','config/metta_generals.ini'):continue
            retained.append((p,rel))
    receipt=dict(schema='generals-row-rotation-result-v1',complete=complete,elapsed_seconds=time.monotonic()-START,
                 files={str(rel):hashlib.sha256(p.read_bytes()).hexdigest() for p,rel in retained})
    (OUTPUT/'collection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with tarfile.open(OUTPUT/'results.tar.gz','w:gz') as archive:
        for p,rel in retained:archive.add(p,arcname='generals/'+str(rel),recursive=False)
    if complete:
        selection=json.loads((RESULTS/'selection.json').read_text())
        (OUTPUT/'COMPLETED.json').write_text(json.dumps(dict(schema='generals-row-rotation-completed-v1',
            selected=selection['selected'],results_sha256=hashlib.sha256((OUTPUT/'results.tar.gz').read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256((RESULTS/'selection.json').read_bytes()).hexdigest(),
            plan_sha256=hashlib.sha256((RESULTS/'plan.json').read_bytes()).hexdigest()))+'\n')

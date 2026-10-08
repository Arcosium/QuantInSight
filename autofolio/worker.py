"""Bounded local genomic search. Queue and state survive worker restarts."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .config import ROOT, RUNS
from .store import initialize, connect, setting, set_setting, event
from .genetics import generate, model_fingerprint
from .catalogue import refresh, rows


def process_is_job(pid,identity):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return b'autofolio.runner' in command and identity.encode() in command
    except (OSError,TypeError):return False


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    return 0


def completed_genomes():
    with connect() as db:
        found=db.execute("SELECT s.payload,j.id job_id FROM strategies s JOIN jobs j ON s.source=j.result WHERE j.status='done'").fetchall()
    return [dict(json.loads(r['payload']),job_id=r['job_id']) for r in found
            if json.loads(r['payload']).get('genome')]


def storage_usage():
    total=0
    for folder,dirs,files in os.walk(RUNS):
        for name in files:
            try:total+=(Path(folder)/name).stat().st_size
            except OSError:pass
    return total


def main():
    initialize()
    from . import research
    research.initialize()
    with connect() as db:db.execute("UPDATE alpha_candidates SET status='queued',message='중단된 평가 재개 대기' WHERE status='running'")
    children={}
    last_refresh=time.time()
    importer=None
    event('worker_started',dict(pid=os.getpid(),concurrency=2,cloud=False))
    while True:
        now=time.time()
        for identity,process in list(children.items()):
            if process.poll() is not None:
                with connect() as db:
                    db.execute("UPDATE jobs SET status='failed',finished=?,error=? WHERE id=? AND status='running'",
                               (now,f'Child exit {process.returncode}; inspect run log',identity))
                del children[identity]
        with connect() as db:
            running=db.execute("SELECT id,pid FROM jobs WHERE status='running'").fetchall()
            for r in running:
                if not process_is_job(r['pid'],r['id']):
                    db.execute("UPDATE jobs SET status='failed',finished=?,error='Interrupted child; not silently retried' WHERE id=?",(now,r['id']))
            running=[r for r in running if process_is_job(r['pid'],r['id'])]
        set_setting('heartbeat',dict(timestamp=now,pid=os.getpid(),running=len(running),
                                     available_memory_gb=round(memory_available()/2**30,1)))
        if now-last_refresh>600 and (importer is None or importer.poll() is not None):
            with (RUNS/'catalogue.log').open('a') as log:
                importer=subprocess.Popen([sys.executable,'-c','from autofolio.catalogue import refresh; print(refresh(),flush=True)'],
                                          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            last_refresh=now
        try:
            market_running=research.tick(max(0,min(8,int(setting('concurrency',2)))-len(running)))
        except Exception as exc:
            event('failed',dict(message='연구 대기열 점검 실패',error=str(exc)[:200]))
            market_running=len(research._PROCESSES)+len(research._INPUT_PROCESSES)
        set_setting('market_running',market_running)
        # Model/result parity is an evidence gate, not an elapsed-time approval.
        if not setting('enabled',True) or not setting('baseline_verified'):
            time.sleep(10)
            continue
        capacity=max(0,min(8,int(setting('concurrency',2)))-len(running)-market_running)
        if capacity:
            import shutil
            usage=storage_usage()
            blocked=usage>int(setting('storage_budget_gb',8))*2**30 or shutil.disk_usage(RUNS).free<20*2**30
            set_setting('storage_guard',dict(blocked=blocked,used_gb=round(usage/2**30,3),budget_gb=setting('storage_budget_gb',8)))
            if blocked:
                time.sleep(10)
                continue
        if capacity and memory_available()>12*2**30:
            with connect() as db:
                waiting=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT ?",(capacity,)).fetchall()
                seen={r['id'] for r in db.execute('SELECT id FROM jobs')}
            if len(waiting)<capacity:
                generation=int(setting('generation',0))+1
                parents=completed_genomes()
                for g in setting('seed_genomes',[]):
                    parents.append(dict(genome=g,job_id='local_ai',net_return=0,negative_months=0,rule_screen_pass=True))
                proposals=generate(parents,seen,capacity-len(waiting),generation)
                with connect() as db:
                    for p in proposals:
                        db.execute("INSERT OR IGNORE INTO jobs(id,genome,model_id,generation,parents,operator,status,created) VALUES (?,?,?,?,?,?,'queued',?)",
                                   (p['id'],json.dumps(p['genome']),model_fingerprint(p['genome'],'pending'),generation,
                                    json.dumps(p['parents']),p['operator'],now))
                set_setting('generation',generation)
            with connect() as db:
                waiting=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT ?",(capacity,)).fetchall()
            for job in waiting:
                directory=RUNS/'experiments'/job['id']
                directory.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
                # Children inherit the service's 4-core/8-GiB slice cap and low priority.
                # A per-child address-space cap would count shared file mappings too;
                # aggregate cgroup memory protects the host without that distortion.
                with (directory/'run.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-m','autofolio.runner','--job',job['id']],
                                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with connect() as db:
                    db.execute("UPDATE jobs SET status='running',pid=?,started=? WHERE id=?",(child.pid,time.time(),job['id']))
                children[job['id']]=child
                event('job_started',dict(id=job['id'],operator=job['operator'],generation=job['generation']))
        time.sleep(10)


if __name__=='__main__':main()

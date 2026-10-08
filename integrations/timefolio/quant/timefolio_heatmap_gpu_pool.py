"""Three bounded Runpod shards; local credentials, recovery and owned-pod cleanup."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

from quant.timefolio_heatmap_gpu_worker import digest, verify, write
from quant.timefolio_heatmap_remote import api, IMAGE

CANDIDATES = [('NVIDIA RTX A5000', 'COMMUNITY'), ('NVIDIA GeForce RTX 3090', 'COMMUNITY'),
              ('NVIDIA GeForce RTX 4090', 'COMMUNITY'), ('NVIDIA RTX A5000', 'SECURE')]
RATE_LIMIT = .40
SECONDS_LIMIT = 7200


def verify_results(folder, manifest_sha, seed):
    folder = Path(folder)
    complete = json.loads((folder/'complete.json').read_text())
    runtime = json.loads((folder/'runtime.json').read_text())
    assert complete['seed'] == seed and complete['folds'] == 54
    assert complete['package_sha256'] == runtime['package_sha256'] == manifest_sha
    assert runtime['device'] == 'cuda' and runtime['gpu']
    hashes = json.loads((folder/'artifact_hashes.json').read_text()); assert len(hashes) == 162
    for name, sha in hashes.items():
        path = folder/name
        assert path.resolve().is_relative_to(folder.resolve()) and digest(path) == sha
    return dict(folds=54, artifacts=162, gpu=runtime['gpu'], seed=seed)


def unpack_results(archive, destination):
    destination = Path(destination); destination.mkdir()
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if sum(m.size for m in members) > 1024**3: raise RuntimeError('Unexpectedly large results')
        for member in members:
            target = destination/member.name
            if (not target.resolve().is_relative_to(destination.resolve()) or
                    not (member.isfile() or member.isdir()) or not member.name.startswith('out')):
                raise RuntimeError('Unsafe result archive member')
        stream.extractall(destination, members=members, filter='data')


def terminate_owned(rest, record, receipt):
    """Idempotent and safe for a concurrent deadline watchdog."""
    remaining = rest('GET','/pods')
    owned = next((p for p in remaining if p['id'] == record['id']), None)
    if owned:
        if owned.get('name') != record['name']: raise RuntimeError('Ownership mismatch')
        rest('DELETE','/pods/'+record['id'])
    for _ in range(4):
        if not any(p['id'] == record['id'] for p in rest('GET','/pods')): break
        time.sleep(2)
    else: raise RuntimeError('Pod deletion not confirmed')
    record.update(state='terminated', deletion_verified=True, finished_at=time.time())
    record['estimated_cost_usd'] = (record['finished_at']-record['started_at'])/3600 * record.get('estimated_hourly',RATE_LIMIT)
    write(receipt, record)


def watchdog(receipt):
    receipt = Path(receipt); record = json.loads(receipt.read_text()); rest = api()
    while time.time() < record['deadline']:
        latest = json.loads(receipt.read_text())
        if latest.get('deletion_verified'): return
        time.sleep(min(15,max(.1,record['deadline']-time.time())))
    # Only these task-owned ephemeral research Pods may be removed.
    for _ in range(6):
        try:
            latest = json.loads(receipt.read_text()); latest['deadline_cleanup'] = True
            terminate_owned(rest, latest, receipt); return
        except Exception:
            time.sleep(10)
    write(receipt.with_name('cleanup_failed.json'),dict(id=record['id'],name=record['name'],requires_reconciliation=True))


def shard(root, seed):
    root=Path(root); folder=root/('pod_seed'+str(seed)); folder.mkdir()
    receipt=folder/'receipt.json'; key=root/'research_key'; rest=api()
    name='arctrade-dailygpu-'+root.name+'-seed'+str(seed)
    if any(p.get('name')==name for p in rest('GET','/pods')): raise RuntimeError('Existing owned Pod must be reconciled')
    record=None; connection=None; errors=[]; ssh=None; scp=None
    try:
        for gpu,cloud in CANDIDATES:
            started=time.time()
            try:
                pod=rest('POST','/pods',dict(name=name,imageName=IMAGE,cloudType=cloud,
                    gpuCount=1,gpuTypeIds=[gpu],containerDiskInGb=20,volumeInGb=0,
                    ports=['22/tcp'],supportPublicIp=True,env={'PUBLIC_KEY':key.with_suffix('.pub').read_text().strip()},
                    minRAMPerGPU=8,minVCPUPerGPU=2))
            except Exception as exc:
                # POST may have succeeded even if its response was lost.
                matches=[p for p in rest('GET','/pods') if p.get('name')==name]
                if len(matches)>1: raise RuntimeError('Ambiguous owned Pod creation')
                if not matches:
                    errors.append(dict(gpu=gpu,cloud=cloud,error=str(exc)[:350]));continue
                pod=matches[0]
            rate=pod.get('costPerHr')
            record=dict(id=pod['id'],name=name,gpu=gpu,cloud=cloud,seed=seed,started_at=started,
                deadline=started+SECONDS_LIMIT,state='starting',quoted_hourly=rate,
                estimated_hourly=float(rate or RATE_LIMIT)+.01,attempts=errors)
            write(receipt,record)
            subprocess.Popen([sys.executable,'-m','quant.timefolio_heatmap_gpu_pool','watchdog','--root',str(receipt)],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            if not rate or record['estimated_hourly']>RATE_LIMIT: raise RuntimeError('Pod quote exceeds hourly ceiling')
            break
        if record is None:
            write(folder/'unavailable.json',dict(attempts=errors,pods_created=0,cost_usd=0))
            return dict(seed=seed,state='unavailable',attempts=len(errors))
        common=['-i',str(key),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new',
                '-o','UserKnownHostsFile='+str(folder/'known_hosts'),'-o','ConnectTimeout=10']
        for _ in range(100):
            pod=rest('GET','/pods/'+record['id']); ip=pod.get('publicIp'); port=(pod.get('portMappings') or {}).get('22')
            if ip and port:
                ssh=['ssh',*common,'-p',str(port),'root@'+ip]
                scp=['scp',*common,'-P',str(port)]
                probe=subprocess.run([*ssh,'true'],capture_output=True,timeout=15)
                if probe.returncode==0: connection=(ip,port);break
            time.sleep(5)
        if not connection: raise TimeoutError('Pod SSH did not become ready')
        ip,port=connection
        subprocess.run([*scp,str(root/'package.tar.gz'),'root@'+ip+':/tmp/heatmap.tar.gz'],check=True,capture_output=True,timeout=240)
        setup="mkdir -p /workspace/heatmap && cd /workspace/heatmap && tar -xzf /tmp/heatmap.tar.gz && python3 -m pip install --break-system-packages -q 'numpy<3' 'scipy<2'"
        subprocess.run([*ssh,setup],check=True,capture_output=True,timeout=240)
        manifest_sha=digest(root/'package/manifest.json')
        remote_hash=subprocess.run([*ssh,"sha256sum /workspace/heatmap/manifest.json"],check=True,capture_output=True,text=True,timeout=30).stdout.split()[0]
        if remote_hash!=manifest_sha:raise AssertionError('Transferred manifest changed')
        command=f"OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 timeout 6600 python3 worker.py --root . --out out --seed {seed} --device cuda > train.log 2>&1; echo $? > exit_code"
        launch="cd /workspace/heatmap && (nohup bash -c "+shlex.quote(command)+" < /dev/null > launch.log 2>&1 &)"
        subprocess.run([*ssh,launch],check=True,capture_output=True,timeout=30)
        record.update(state='training',manifest_sha256=manifest_sha);write(receipt,record)
        while time.time()<record['deadline']-180:
            check=subprocess.run([*ssh,"cd /workspace/heatmap; if test -f exit_code; then cat exit_code; else echo RUNNING; fi"],capture_output=True,text=True,timeout=30)
            if check.returncode==0 and check.stdout.strip()!='RUNNING':
                record['training_exit']=check.stdout.strip();break
            progress=subprocess.run([*ssh,"cd /workspace/heatmap; test ! -f out/progress.json || cat out/progress.json"],capture_output=True,text=True,timeout=30)
            if progress.stdout.strip():
                value=json.loads(progress.stdout);write(folder/'progress.json',value)
                print(json.dumps(dict(seed=seed,**value)),flush=True)
            time.sleep(20)
        else:record['training_exit']='deadline'
    except Exception as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            # Keep stderr: the exception string alone can omit the actual failure.
            stderr=exc.stderr.decode(errors='replace') if isinstance(exc.stderr,bytes) else (exc.stderr or '')
            stdout=exc.stdout.decode(errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or '')
            write(folder/'command_failure.json',dict(returncode=exc.returncode,stderr=stderr[-5000:],stdout=stdout[-2000:]))
        if record is not None:record['error']=type(exc).__name__+': '+str(exc)[:350]
        else:write(folder/'error.json',dict(error=type(exc).__name__+': '+str(exc)[:350]))
    finally:
        if record is not None:
            # Recover before deletion, including incomplete output/logs after failure.
            if connection:
                try:
                    export="cd /workspace/heatmap && mkdir -p out && cp train.log out/train.log && tar -czf /tmp/heatmap-results.tar.gz out"
                    subprocess.run([*ssh,export],check=True,capture_output=True,timeout=90)
                    subprocess.run([*scp,'root@'+connection[0]+':/tmp/heatmap-results.tar.gz',str(folder/'results.tar.gz')],check=True,capture_output=True,timeout=120)
                    unpack_results(folder/'results.tar.gz',folder/'recovered');record['recovered']=True
                    if record.get('training_exit')=='0':
                        record['result_validation']=verify_results(folder/'recovered/out',digest(root/'package/manifest.json'),seed)
                except Exception as exc:record['recovery_error']=type(exc).__name__+': '+str(exc)[:350]
            write(receipt,record)
            terminate_owned(rest,record,receipt)
    return dict(seed=seed,**({k:record.get(k) for k in ['state','training_exit','deletion_verified','estimated_cost_usd','error','recovery_error','result_validation']} if record else {'state':'error'}))


def run(root):
    root=Path(root);verify(root/'package')
    receipt=json.loads((root/'export_receipt.json').read_text())
    if digest(root/'package.tar.gz')!=receipt['archive_sha256']:raise AssertionError('Export changed')
    if (root/'pool_start.json').exists():raise RuntimeError('Pool already started')
    key=root/'research_key'
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(key)],check=True)
    write(root/'pool_start.json',dict(started_at=time.time(),seeds=[17,29,43],max_hourly=1.2,max_hours_per_pod=2))
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs=[executor.submit(shard,root,seed) for seed in [17,29,43]]
        results=[]
        for seed,job in zip([17,29,43],jobs):
            try:results.append(job.result())
            except Exception as exc:results.append(dict(seed=seed,state='controller_error',error=str(exc)[:350]))
    write(root/'pool_complete.json',dict(shards=results,complete=all(r.get('result_validation') for r in results),
        estimated_cost_usd=sum(r.get('estimated_cost_usd',0) or 0 for r in results)))
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','watchdog'])
    parser.add_argument('--root',type=Path,required=True);a=parser.parse_args()
    if a.action=='run':run(a.root)
    else:watchdog(a.root)

"""One renewable research Pod; retain it across batches and recover each job."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
from quant.timefolio_heatmap_gpu_worker import digest,verify,write
from quant.timefolio_heatmap_remote import api,IMAGE
from quant.timefolio_heatmap_fleet_lab import MAX_PODS

CANDIDATES=[('NVIDIA GeForce RTX 3090','COMMUNITY'),('NVIDIA GeForce RTX 4090','COMMUNITY'),
    ('NVIDIA RTX A5000','SECURE'),('NVIDIA L4','SECURE'),('NVIDIA A40','SECURE')]
MAX_HOURLY=.50


def read(path):return json.loads(Path(path).read_text())


def stop(root,reason):
    record=read(root/'pod.json');rest=api()
    pod=next((p for p in rest('GET','/pods') if p['id']==record['id']),None)
    if pod:
        if pod['name']!=record['name']:raise RuntimeError('Pod ownership mismatch')
        rest('DELETE','/pods/'+pod['id'])
    if any(p['id']==record['id'] for p in rest('GET','/pods')):raise RuntimeError('Deletion unverified')
    write(root/'deletion.json',dict(id=record['id'],name=record['name'],reason=reason,deletion_verified=True,
        finished_at=time.time(),estimated_cost_usd=(time.time()-record['started_at'])/3600*record['estimated_hourly']))


def watchdog(root):
    while not (root/'deletion.json').exists():
        if time.time()>=read(root/'lease.json')['expires_at']:
            for _ in range(6):
                try:stop(root,'lease_expired');return
                except Exception:time.sleep(10)
            write(root/'cleanup_failed.json',dict(requires_reconciliation=True));return
        time.sleep(15)


def renew(root):
    if (root/'deletion.json').exists():raise RuntimeError('Pod already deleted')
    write(root/'lease.json',dict(renewed_at=time.time(),expires_at=time.time()+6*3600,max_hours=6))


def connection(root):
    p=read(root/'pod.json');common=['-i',str(root/'research_key'),'-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(root/'known_hosts'),'-o','ConnectTimeout=10']
    return (['ssh',*common,'-p',str(p['port']),'root@'+p['ip']],['scp',*common,'-P',str(p['port'])],p['ip'])


def command(args,timeout=30):
    return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout).stdout


def rent(root):
    if root.exists():raise RuntimeError('Fresh Pod root required; reuse with submit or monitor')
    root.mkdir();key=root/'research_key'
    subprocess.run(['ssh-keygen','-q','-t','ed25519','-N','','-f',str(key)],check=True)
    rest=api();name='arctrade-lab-'+root.name;errors=[];record=None
    existing=rest('GET','/pods')
    if any(p.get('name')==name for p in existing):raise RuntimeError('Existing owned Pod must be reconciled')
    if len(existing)>=MAX_PODS:raise RuntimeError('Runpod concurrent Pod limit reached')
    for gpu,cloud in CANDIDATES:
        started=time.time()
        try:
            pod=rest('POST','/pods',dict(name=name,imageName=IMAGE,cloudType=cloud,gpuCount=1,gpuTypeIds=[gpu],
                allowedCudaVersions=['12.8','12.9','13.0'],containerDiskInGb=20,volumeInGb=0,ports=['22/tcp'],
                supportPublicIp=True,env={'PUBLIC_KEY':key.with_suffix('.pub').read_text().strip()},minRAMPerGPU=16,minVCPUPerGPU=4))
        except Exception as exc:
            matches=[p for p in rest('GET','/pods') if p.get('name')==name]
            if len(matches)>1:raise RuntimeError('Ambiguous creation')
            if not matches:
                errors.append(dict(gpu=gpu,cloud=cloud,error=str(exc)[:350]));continue
            pod=matches[0]
        record=dict(id=pod['id'],name=name,gpu=gpu,cloud=cloud,started_at=started,quoted_hourly=pod.get('costPerHr'),
            estimated_hourly=float(pod.get('costPerHr') or MAX_HOURLY)+.01,attempts=errors)
        write(root/'pod.json',record);renew(root)
        subprocess.Popen([sys.executable,'-m','quant.timefolio_heatmap_lab_pod','watchdog','--pod-root',str(root)],
            stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        if record['estimated_hourly']>MAX_HOURLY:
            stop(root,'quote_exceeds_ceiling');raise RuntimeError('Hourly quote exceeds ceiling')
        break
    if record is None:
        write(root/'unavailable.json',dict(attempts=errors));raise RuntimeError('No eligible GPU available')
    try:
        for _ in range(120):
            pod=rest('GET','/pods/'+record['id']);ip=pod.get('publicIp');port=(pod.get('portMappings') or {}).get('22')
            if ip and port:
                record.update(ip=ip,port=port);write(root/'pod.json',record)
                ssh,_,_=connection(root)
                try:command([*ssh,'true'],15);break
                except (subprocess.SubprocessError,OSError):pass
            time.sleep(5)
        else:raise TimeoutError('SSH startup exceeded10minutes')
        command([*ssh,"python3 -m pip install --break-system-packages -q 'numpy<3' 'scipy<2'"],240)
        write(root/'ready.json',dict(ready_at=time.time(),id=record['id'],quoted_hourly=record['quoted_hourly']))
        print(json.dumps(dict(state='ready',gpu=record['gpu'],quoted_hourly=record['quoted_hourly'])),flush=True)
    except Exception:
        stop(root,'startup_failed');raise


def validate_job(folder,plan,manifest_sha,name):
    jobs={row['id']:row for row in plan['jobs']}
    if name not in jobs:raise ValueError('Unregistered job')
    result=read(folder/'complete.json');runtime=read(folder/'runtime.json');hashes=read(folder/'artifact_hashes.json')
    cfg=next(c for c in plan['cases'] if c['id']==jobs[name]['case'])
    assert result['folds']==9 and result['model']==name
    assert result['package_sha256']==runtime['package_sha256']==manifest_sha
    assert runtime['device']=='cuda' and runtime['gpu'] and runtime['config']['seed']==jobs[name]['seed']
    assert all(runtime['config'][k]==v for k,v in cfg.items() if k!='id')
    expected={f'models/{name}/{fold["month"]}{suffix}' for fold in plan['folds'] for suffix in ['.pt','.json','.pred.npy']}
    expected|={'untrained.pt','untrained.pred.npy'}
    assert set(hashes)==expected
    for path,sha in hashes.items():
        target=folder/path
        assert target.resolve().is_relative_to(folder.resolve()) and not target.is_symlink() and digest(target)==sha
    return dict(id=name,folds=9,artifacts=len(hashes),gpu=runtime['gpu'])


def recover(pod_root,study,names):
    if not names:return
    plan=read(study/'package/plan.json');registered={j['id'] for j in plan['jobs']}
    assert set(names)<=registered and all(re.fullmatch('[a-z0-9_]+',n) for n in names)
    ssh,scp,ip=connection(pod_root);remote='/workspace/'+study.name
    archive='recovery_'+str(time.time_ns())+'.tar.gz';local=study/archive
    files=' '.join(shlex.quote('jobs/'+n) for n in names)
    command([*ssh,f'cd {shlex.quote(remote)} && tar -czf /tmp/{archive} {files}'],120)
    command([*scp,'root@'+ip+':/tmp/'+archive,str(local)],180)
    target=study/'recovered';target.mkdir(exist_ok=True)
    with tarfile.open(local) as stream:
        members=stream.getmembers()
        if sum(m.size for m in members)>1024**3:raise RuntimeError('Excessive result size')
        for m in members:
            p=Path(m.name)
            if len(p.parts)<2 or p.parts[:2] not in {('jobs',n) for n in names} or not (m.isfile() or m.isdir()) or not (target/p).resolve().is_relative_to(target.resolve()):
                raise RuntimeError('Unsafe result archive')
            if (target/p).exists():raise RuntimeError('Recovery would overwrite existing output')
        stream.extractall(target,members=members,filter='data')
    reviews=[validate_job(target/'jobs'/name,plan,digest(study/'package/manifest.json'),name) for name in names]
    journal=study/'recovered_jobs.json';old=read(journal) if journal.exists() else {}
    old.update({row['id']:row for row in reviews});write(journal,old)
    local.unlink();command([*ssh,'rm -- /tmp/'+shlex.quote(archive)])


def submit(pod_root,study):
    verify(study/'package');export=read(study/'export_receipt.json')
    assert digest(study/'package.tar.gz')==export['archive_sha256']
    if (study/'submission.json').exists():raise RuntimeError('Already submitted; use monitor')
    if (pod_root/'deletion.json').exists():raise RuntimeError('Pod deleted')
    ssh,scp,ip=connection(pod_root);remote='/workspace/'+study.name
    if not re.fullmatch('[a-zA-Z0-9_]+',study.name):raise ValueError('Unsafe study identifier')
    command([*scp,str(study/'package.tar.gz'),'root@'+ip+':/tmp/'+study.name+'.tar.gz'],240)
    command([*ssh,f'mkdir {remote} && cd {remote} && tar -xzf /tmp/{study.name}.tar.gz'],60)
    sha=command([*ssh,'sha256sum '+remote+'/manifest.json']).split()[0]
    assert sha==export['manifest_sha256']
    shell='OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 scheduler.py --root . > queue.log 2>&1; echo $? > queue_exit'
    launch=f'cd {remote} && (nohup bash -c '+shlex.quote(shell)+' < /dev/null > launch.log 2>&1 &)'
    command([*ssh,launch]);renew(pod_root)
    write(study/'submission.json',dict(pod_id=read(pod_root/'pod.json')['id'],pod_root=str(pod_root),
        submitted_at=time.time(),manifest_sha256=sha,parallel_jobs=2,terminate_on_completion=False))


def monitor(pod_root,study):
    ssh,_,_=connection(pod_root);remote='/workspace/'+study.name
    while not (pod_root/'deletion.json').exists():
        try:
            output=command([*ssh,f'cd {remote} && if test -f queue_status.json; then cat queue_status.json; else echo null; fi'])
            status=json.loads(output)
            if status:
                write(study/'remote_status.json',status)
                recovered=read(study/'recovered_jobs.json') if (study/'recovered_jobs.json').exists() else {}
                recover(pod_root,study,[n for n in status['completed'] if n not in recovered])
                print(json.dumps({k:len(status[k]) if isinstance(status[k],list) else status[k] for k in ['state','completed','running','failed','pending']}),flush=True)
                if status['state'] in ['complete','failed']:
                    if status['state']=='failed' and status['running']:time.sleep(10);continue
                    break
            else:
                code=command([*ssh,f'cd {remote} && if test -f queue_exit; then cat queue_exit; else echo RUNNING; fi']).strip()
                if code!='RUNNING':raise RuntimeError('Remote queue exited before status, code='+code)
        except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
            write(study/'transient_monitor_error.json',dict(at=time.time(),type=type(exc).__name__))
        time.sleep(20)
    for file in ['queue.log','gpu_smoke.json']:
        try:(study/file).write_text(command([*ssh,'cat '+remote+'/'+file]))
        except subprocess.SubprocessError:pass
    write(study/'monitor_finished.json',dict(at=time.time(),pod_retained=not (pod_root/'deletion.json').exists(),
        recovered=len(read(study/'recovered_jobs.json')) if (study/'recovered_jobs.json').exists() else 0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','submit','monitor','renew','watchdog','stop'])
    p.add_argument('--pod-root',type=Path,required=True);p.add_argument('--study',type=Path);a=p.parse_args()
    if a.action=='watchdog':watchdog(a.pod_root)
    elif a.action=='renew':renew(a.pod_root)
    elif a.action=='stop':stop(a.pod_root,'explicit_task_cleanup')
    else:
        if a.study is None:raise ValueError('Study required')
        if a.action=='run':
            verify(a.study/'package');rent(a.pod_root)
        if a.action in ['run','submit']:submit(a.pod_root,a.study)
        monitor(a.pod_root,a.study)

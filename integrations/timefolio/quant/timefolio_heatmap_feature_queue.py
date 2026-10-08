"""Portable two-process queue; fixed cases, no outcome-dependent job selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import engine
from models import LabNet


def smoke(root):
    engine.verify(root);engine.configure('cuda')
    plan=json.loads((root/'plan.json').read_text())
    rows=[]
    for cfg in plan['cases']:
        raw=np.load(root/f"images_{cfg['encoding']}.npy",mmap_mode='r')[:200]
        x=torch.tensor(np.array(raw),device='cuda').float().div(127.5).sub(1)
        states=[];tick=time.monotonic()
        for _ in range(2):
            torch.manual_seed(17);model=LabNet(cfg).cuda()
            out=model(x);loss=out.square().mean();loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError('Nonfinite CUDA smoke result')
            states.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone())
        if not torch.equal(states[0],states[1]):raise AssertionError('CUDA gradient reproduction failed')
        rows.append(dict(case=cfg['id'],seconds=time.monotonic()-tick,reproduced=True))
        del model,states
    engine.write(root/'gpu_smoke.json',dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cases=rows))


def run(root):
    root=root.resolve();os.chdir(root);smoke(root)
    plan=json.loads((root/'plan.json').read_text());jobs=plan['jobs'];active={};done=[];failed=[];next_job=0
    status=root/'queue_status.json'
    def save():
        engine.write(status,dict(completed=done,failed=failed,running=list(active),pending=len(jobs)-next_job,
            expected=len(jobs),updated_at=time.time(),state='failed' if failed else ('complete' if len(done)==len(jobs) else 'running')))
    (root/'jobs').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    while next_job<len(jobs) or active:
        while not failed and next_job<len(jobs) and len(active)<2:
            job=jobs[next_job];next_job+=1;name=job['id']
            if (root/'jobs'/name).exists():raise RuntimeError('Immutable job output already exists')
            log=(root/'logs'/(name+'.log')).open('x')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            process=subprocess.Popen([sys.executable,'worker.py','--root',str(root),'--out',str(root/'jobs'/name),
                '--case',job['case'],'--seed',str(job['seed'])],stdout=log,stderr=subprocess.STDOUT,env=env)
            active[name]=(process,log);save()
        for name,(process,log) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close();del active[name]
                if code==0 and (root/'jobs'/name/'complete.json').exists():done.append(name)
                else:failed.append(dict(id=name,exit_code=code))
                save()
        if failed and not active:break
        save();time.sleep(5)
    save()
    if failed:raise RuntimeError('Queue stopped after failed job; completed results are preserved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)

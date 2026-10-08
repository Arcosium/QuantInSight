"""Bounded remote GPU queue with shared candidate-stop and configuration-only jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from quant.timefolio_chart_lab import cases, SEEDS
from quant.timefolio_heatmap_gpu_worker import verify, digest, write


def run(root, shard):
    root=root.resolve(); os.chdir(root); verify(root)
    plan=json.loads((root/'plan.json').read_text()); known={c['id']:c for c in cases()}
    assert plan['cases']==cases()
    assert json.loads((root/'registration_main_review.json').read_text())['manifest_sha256']==digest(root/'manifest.json')
    parity=json.loads((root/'remote_parity.json').read_text())
    assert parity['status']=='remote_parity_passed' and parity['manifest_sha256']==digest(root/'manifest.json')
    assert json.loads((root/'remote_tests.json').read_text())['passed']
    for name in ['jobs','logs','accounts','candidates']:(root/name).mkdir(exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    assigned=[]; jobs=[]; active={}; evaluating={}; preparing={}; prepared=set()
    done=[]; accounts=[]; failed=[]; stopped=[]; cursor=0
    def launch(args,logname):
        log=(root/'logs'/logname).open('x'); proc=subprocess.Popen([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
        return proc,log
    while True:
        inbox=json.loads((root/'inbox.json').read_text()); incoming=inbox['cases']
        assert inbox['shard']==shard and incoming[:len(assigned)]==assigned
        assert len(incoming)==len(set(incoming)) and set(incoming)<=set(known)
        for case in incoming[len(assigned):]:jobs.extend(dict(id=f'{case}_seed{s}',case=case,seed=s) for s in SEEDS)
        assigned=incoming; closed=inbox['closed']; halt=(root/'STOP.json').exists()
        for view,(proc,log) in list(preparing.items()):
            if proc.poll() is not None:
                log.close(); del preparing[view]
                if proc.returncode==0 and (root/f'images_{view}.json').exists(): prepared.add(view)
                else: failed.append(dict(stage='images',id=view,exit_code=proc.returncode))
        if not halt and not failed and not preparing:
            needed=[known[j['case']]['view'] for j in jobs[cursor:]]
            missing=next((v for v in needed if v not in prepared),None)
            if missing:preparing[missing]=launch(['-m','quant.timefolio_chart_images','--root',str(root),'--view',missing],'images_'+missing+'.log')
        while not halt and not failed and cursor<len(jobs) and len(active)<2:
            job=jobs[cursor]
            if known[job['case']]['view'] not in prepared:break
            cursor+=1
            active[job['id']]=launch(['-m','quant.timefolio_chart_worker','--root',str(root),'--out',str(root/'jobs'/job['id']),
                '--case',job['case'],'--seed',str(job['seed'])],job['id']+'.log')
        for name,(proc,log) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[name]
                if (root/'jobs'/name/'stopped.json').exists():stopped.append(name)
                elif proc.returncode==0 and (root/'jobs'/name/'complete.json').exists():done.append(name)
                else:failed.append(dict(stage='training',id=name,exit_code=proc.returncode))
        candidates=(['online_nonimage'] if inbox.get('online',False) else [])+assigned
        for case in candidates:
            if halt or failed or evaluating:break
            if case in accounts or case in evaluating:continue
            if case!='online_nonimage' and not all(f'{case}_seed{s}' in done for s in SEEDS):continue
            evaluating[case]=launch(['-m','quant.timefolio_chart_accounts','--root',str(root),'--case',case],'accounts_'+case+'.log')
        for case,(proc,log) in list(evaluating.items()):
            if proc.poll() is not None:
                log.close(); del evaluating[case]
                if (root/'accounts'/case/'stopped.json').exists():stopped.append(case)
                elif proc.returncode==0 and (root/'accounts'/case/'complete.json').exists():accounts.append(case)
                else:failed.append(dict(stage='accounts',id=case,exit_code=proc.returncode))
        ended=(closed and len(accounts)==len(candidates)) or ((halt or failed) and not active and not evaluating and not preparing)
        state='failed' if failed else 'stopped' if halt else 'complete' if ended else 'running'
        write(root/'queue_status.json',dict(shard=shard,pid=os.getpid(),updated_at=time.time(),state=state,
            assigned_cases=assigned,pending=len(jobs)-cursor,expected=len(jobs),completed=done,running=list(active),
            accounts_complete=accounts,accounts_running=list(evaluating),images_running=list(preparing),
            failed=failed,stopped=stopped,inbox_closed=closed,gpu_parallel_jobs=2,cpu_parallel_jobs=1))
        if ended:break
        time.sleep(2)
    if failed:raise RuntimeError('Retain outputs for main failure review')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--shard',type=int,required=True)
    a=p.parse_args(); run(a.root,a.shard)

"""Bounded isolated GPU worker queue; no cloud credentials or broker access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def policy_mode(job, *, account_panel):
    mode = job.get('policy_selection', 'fixed')
    if mode not in ('fixed', 'retention_grid'):
        raise ValueError('Unregistered account policy selection')
    if mode == 'retention_grid' and (not account_panel or job.get('rebalance') != 5
                                      or job.get('rank_buffer') != 5):
        raise ValueError('Retention selection requires an account panel, R5 and buffer5 reference')
    return mode


def job_command(spec, job):
    mode = policy_mode(job, account_panel=spec.get('account_panel'))
    cmd = [sys.executable, '-u', '-m', 'quant.timefolio_cnn_train',
        '--dataset', spec['dataset'], '--output', str(Path(spec['output'])/job['name']),
        '--fold', job['fold'], '--objective', job['objective'],
        '--seed', str(job['seed']), '--device', 'cuda', '--purpose', 'exploratory']
    if spec.get('account_panel'):
        cmd += ['--account-panel', spec['account_panel'],
                '--rebalance', str(job['rebalance']), '--rank-buffer', str(job['rank_buffer'])]
    if mode != 'fixed':
        cmd += ['--policy-selection', mode]
    return cmd


def run(config):
    spec = json.loads(Path(config).read_text())
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    jobs = list(spec['jobs']); active = {}; done = []
    while jobs or active:
        while jobs and len(active) < spec.get('parallel', 2):
            job = jobs.pop(0); name = job['name']
            log = (output/(name+'.log')).open('w')
            cmd = job_command(spec, job)
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            active[name] = (process, log, time.time())
        for name, (process, log, began) in list(active.items()):
            code = process.poll()
            if code is not None:
                log.close(); done.append(dict(name=name, exit_code=code, seconds=time.time()-began))
                del active[name]
        progress = dict(at=time.time(), active=list(active), pending=len(jobs), completed=done)
        temp = output/'progress.tmp'; temp.write_text(json.dumps(progress, indent=2)+'\n'); temp.replace(output/'progress.json')
        if jobs or active:
            time.sleep(5)
    (output/'complete.json').write_text(json.dumps(progress, indent=2)+'\n')
    return int(any(job['exit_code'] for job in done))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True)
    sys.exit(run(ap.parse_args().config))

"""Two GPU jobs plus bounded CPU accounts; failures stop new dispatch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from quant.timefolio_heatmap_gpu_worker import verify, write, digest
from quant.timefolio_heatmap_fleet_lab import cases, SEEDS


def run(root, shard):
    root = root.resolve(); os.chdir(root); verify(root)
    plan = json.loads((root / 'plan.json').read_text())
    assigned = []
    known = {c['id'] for c in cases()}
    assert len(assigned) == len(set(assigned)) and set(assigned) <= known
    parity = json.loads((root / 'remote_parity.json').read_text())
    assert parity['status'] == 'remote_parity_passed' and parity['manifest_sha256'] == digest(root / 'manifest.json')
    # A case's three seeds stay on one Pod for ensemble construction.
    jobs = []
    (root / 'jobs').mkdir(); (root / 'logs').mkdir(); (root / 'accounts').mkdir()
    active = {}; evaluating = {}; done = []; accounts = []; failed = []; cursor = 0
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    cpu_workers = 1
    closed = False
    def save():
        write(root / 'queue_status.json', dict(shard=shard, updated_at=time.time(), completed=done,
            running=list(active), failed=failed, pending=len(jobs) - cursor, expected=len(jobs),
            accounts_complete=accounts, accounts_running=list(evaluating), expected_cases=len(assigned),
            assigned_cases=assigned, inbox_closed=closed,
            gpu_parallel_jobs=2, cpu_parallel_jobs=cpu_workers,
            state='failed' if failed else 'complete' if closed and len(accounts) == len(assigned) else 'running'))
    while True:
        inbox = json.loads((root / 'inbox.json').read_text())
        incoming = inbox['cases']
        assert incoming[:len(assigned)] == assigned and len(incoming) == len(set(incoming))
        assert set(incoming) <= known and inbox['shard'] == shard
        for case in incoming[len(assigned):]:
            jobs.extend(dict(id=f'{case}_seed{s}', case=case, seed=s) for s in SEEDS)
        assigned = incoming; closed = inbox['closed']
        while not failed and cursor < len(jobs) and len(active) < 2:
            job = jobs[cursor]; cursor += 1
            log = (root / 'logs' / (job['id'] + '.log')).open('x')
            proc = subprocess.Popen([sys.executable, 'worker.py', '--root', str(root), '--out',
                str(root / 'jobs' / job['id']), '--case', job['case'], '--seed', str(job['seed'])],
                env=env, stdout=log, stderr=subprocess.STDOUT)
            active[job['id']] = (proc, log)
        for name, (proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close(); del active[name]
                if proc.returncode == 0 and (root / 'jobs' / name / 'complete.json').exists(): done.append(name)
                else: failed.append(dict(stage='training', id=name, exit_code=proc.returncode))
        for case in assigned:
            if failed or len(evaluating) >= cpu_workers: break
            if case in accounts or case in evaluating: continue
            if not all(f'{case}_seed{s}' in done for s in SEEDS): continue
            log = (root / 'logs' / ('accounts_' + case + '.log')).open('x')
            proc = subprocess.Popen([sys.executable, '-m', 'quant.timefolio_heatmap_fleet_accounts',
                '--root', str(root), '--case', case], env=env, stdout=log, stderr=subprocess.STDOUT)
            evaluating[case] = (proc, log)
        for case, (proc, log) in list(evaluating.items()):
            if proc.poll() is not None:
                log.close(); del evaluating[case]
                if proc.returncode == 0 and (root / 'accounts' / case / 'complete.json').exists(): accounts.append(case)
                else: failed.append(dict(stage='accounts', id=case, exit_code=proc.returncode))
        save()
        if failed and not active and not evaluating: break
        if closed and len(accounts) == len(assigned): break
        time.sleep(2)
    save()
    if failed: raise RuntimeError('Shard failed; completed artifacts are retained for recovery')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True); args = parser.parse_args()
    run(args.root, args.shard)

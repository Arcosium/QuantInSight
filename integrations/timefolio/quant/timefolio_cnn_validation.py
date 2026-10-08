"""Recover-first continuation: chronological gates and all registered accounts.

Reads one explicitly supplied research lease; it never creates/deletes pods,
changes collectors, or reads other sessions' allocation state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def run(dataset, panel, lease_root, output):
    dataset, panel, lease_root, output = map(Path, [dataset, panel, lease_root, output])
    if output.exists():
        raise ValueError('Use a fresh validation directory')
    output.mkdir(parents=True)

    def state(**values):
        (output/'progress.json').write_text(json.dumps(dict(at=time.time(), **values), indent=2)+'\n')

    state(state='waiting_for_verified_recovery', lease=str(lease_root/'lease.json'))
    try:
        deadline = time.monotonic()+7200
        while time.monotonic() < deadline:
            try:
                lease = json.loads((lease_root/'lease.json').read_text())
            except json.JSONDecodeError:
                time.sleep(2)
                continue
            if lease['state'] == 'released':
                if lease.get('error') or not lease.get('artifacts_recovered'):
                    raise ValueError('Owned GPU lease ended without a verified full recovery')
                if any(j['exit_code'] != 0 for j in lease['completed_jobs']):
                    raise ValueError('Some registered models failed; no silent subset selection')
                break
            time.sleep(20)
        else:
            raise TimeoutError('Recovery was not ready within the registered waiting budget')
        from quant.timefolio_cnn_dataset import sha
        from quant.timefolio_cnn_gate import build
        from quant.timefolio_cnn_account import run_accounts
        if sha(dataset/'manifest.json') != lease['dataset_manifest_sha256']:
            raise ValueError('Lease dataset differs from validation dataset')
        completed = []
        for objective in ['mse', 'bce', 'listnet', 'topk_ce']:
            state(state='building_gate', objective=objective, completed=completed)
            gate = output/('gate_'+objective)
            build(dataset, lease_root/'recovered/results', gate, objective)
            state(state='replaying_accounts', objective=objective, completed=completed)
            run_accounts(panel, gate, output/('accounts_'+objective))
            completed.append(objective)
        state(state='complete', objectives=completed, account_cases=36,
              contest_certified=False, lease_released_before_local_validation=True)
    except Exception as error:
        state(state='failed', error_type=type(error).__name__, message=str(error)[:300])
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ['dataset', 'panel', 'lease_root', 'output']:
        ap.add_argument('--'+name.replace('_', '-'), required=True)
    run(**vars(ap.parse_args()))

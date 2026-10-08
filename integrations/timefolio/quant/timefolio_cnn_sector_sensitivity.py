"""Matched account sensitivity using an observed Timefolio sector-limit snapshot.

The snapshot is applied counterfactually across the research history. It is NOT
a reconstruction of each historical day's sector weights or classifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_account import write


def snapshot_limits(snapshot):
    date = snapshot['requested_date'].replace('-', '')
    if len(date) != 8 or not date.isdigit() or date > '20260923':
        raise ValueError('Sector snapshot crosses the reserved outcome cutoff')
    rows = snapshot['sectors']; codes = [int(r['sector']) for r in rows]
    expected = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}
    weights = np.array([r['market_weight_percent'] for r in rows], dtype=float)
    if (set(codes) != expected or len(codes) != len(expected) or not np.isfinite(weights).all()
            or (weights < 0).any() or abs(weights.sum()-100.) > .1):
        raise ValueError('Eleven unique sectors with percentage weights summing to100 are required')
    return {code: max(.1, 2*float(weight)/100) for code, weight in zip(codes, weights)}


def derive_panel(parent, snapshot_path, output):
    parent, snapshot_path, output = map(Path, [parent, snapshot_path, output])
    if output.exists():
        raise ValueError('Use a fresh sector-sensitivity panel')
    proof = json.loads((parent/'receipt.json').read_text())
    for name, digest in proof['artifacts'].items():
        p = (parent/name).resolve()
        if not p.is_relative_to(parent.resolve()) or sha(p) != digest:
            raise ValueError('Parent account panel changed')
    snapshot = json.loads(snapshot_path.read_text()); limits = snapshot_limits(snapshot)
    with np.load(parent/'panel.npz', allow_pickle=False) as z:
        panel = {name: z[name] for name in z.files}
    for j, sector in enumerate(panel['sector']):
        if sector not in limits and panel['eligible'][j].any():
            raise ValueError('Eligible security has no official snapshot sector')
        panel['sector_cap'][j] = limits.get(int(sector), .1)
    output.mkdir(parents=True)
    np.savez(output/'panel.npz', **panel)
    for filename in ['index.json', 'provenance.json']:
        shutil.copy2(parent/filename, output/filename)
    shutil.copy2(snapshot_path, output/'sector_snapshot.json')
    proof['limitations'] = [v for v in proof['limitations'] if not v.startswith('Uniform10% sector caps')]
    proof['limitations'].append('Official Timefolio sector weights requested for'+snapshot['requested_date']+
        'are held fixed across all historical dates as a counterfactual. Historical daily sector weights are NOT certified.')
    proof.update(parent_receipt_sha256=sha(parent/'receipt.json'), sector_snapshot_sha256=sha(snapshot_path),
        sector_cap_method='max(10%,2*observed_market_weight); fixed historical sensitivity only',
        sector_limit_fractions=limits, historical_sector_weights_certified=False,
        source_sha256=sha(__file__), artifacts={p.name: sha(p) for p in output.iterdir()})
    write(output/'receipt.json', proof)
    return proof


def continue_accounts(dataset, panel, primary_validation, output):
    dataset, panel, primary_validation, output = map(Path, [dataset, panel, primary_validation, output])
    if output.exists():
        raise ValueError('Use a fresh account-sensitivity result directory')
    output.mkdir(parents=True)
    def state(**values):
        write(output/'progress.json', dict(at=time.time(), **values))
    state(state='waiting_for_primary_account_validation')
    try:
        deadline = time.monotonic()+7200
        while time.monotonic() < deadline:
            try:
                status = json.loads((primary_validation/'progress.json').read_text())['state']
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(5); continue
            if status == 'failed':
                raise ValueError('Primary validation failed; do not silently omit registered models')
            if status == 'complete':
                break
            time.sleep(20)
        else:
            raise TimeoutError('Primary account validation was not ready')
        from quant.timefolio_cnn_account import run_accounts
        from quant.timefolio_cnn_controls import run as run_controls
        completed = []
        for objective in ['mse', 'bce', 'listnet', 'topk_ce']:
            state(state='replaying_accounts', objective=objective, completed=completed)
            run_accounts(panel, primary_validation/('gate_'+objective), output/('accounts_'+objective))
            completed.append(objective)
        state(state='replaying_controls', completed=completed)
        run_controls(dataset, panel, output/'controls')
        state(state='complete', objectives=completed, model_account_cases=36, control_cases=8,
              contest_certified=False, historical_sector_weights_certified=False)
    except Exception as error:
        state(state='failed', error_type=type(error).__name__, message=str(error)[:300])
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('panel')
    for name in ['parent', 'snapshot', 'output']:
        p.add_argument('--'+name, required=True)
    p = sub.add_parser('run')
    for name in ['dataset', 'panel', 'primary-validation', 'output']:
        p.add_argument('--'+name, required=True)
    args = ap.parse_args()
    if args.command == 'panel':
        derive_panel(args.parent, args.snapshot, args.output)
    else:
        continue_accounts(args.dataset, args.panel, args.primary_validation, args.output)

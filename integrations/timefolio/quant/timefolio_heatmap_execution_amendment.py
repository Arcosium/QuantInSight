"""Versioned execution amendment on fixed, monthly out-of-sample forecasts.

Historical development data have already informed research choices. Family-wise
bootstrap evidence remains exploratory, not prospective confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_planning_probe import ACTION, announced_action_check
from quant.timefolio_heatmap_replay import INITIAL_NAV
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import (
    market_regimes, load_scores, calendar_blocks, family_bootstrap,
)

DEST = SOURCE.with_name('20260928_execution_amendment_v4')
BUDGETS = [3, 10, 20]
DEPENDENCIES = [
    'execution_amendment', 'planned_replay', 'planned_audit', 'planning_probe',
    'capacity_probe', 'walkforward_eval', 'walkforward', 'replay', 'study',
    'features', 'data',
]


def corrected_panel(p, ix):
    panel = dict(p)
    panel['split'] = p['split'].astype(float).copy()
    panel['split'][ix['codes'].index(ACTION['code']), ix['dates'].index(ACTION['ex_date'])] = ACTION['ratio']
    release = {(ACTION['code'], ACTION['ex_date']): ACTION['listing_date']}
    return panel, release


def freeze(dest=DEST, model_root=MODEL_ROOT):
    dest, model_root = Path(dest), Path(model_root)
    model_spec = json.loads((model_root / 'protocol.json').read_text())
    files = [Path(__file__).with_name(f'timefolio_heatmap_{name}.py') for name in DEPENDENCIES]
    spec = {
        'model_root': str(model_root), 'model_protocol': model_spec,
        'order_budgets': BUDGETS, 'planning_price': 'open',
        'execution': '09:05-09:34 proxy VWAP; 5% window volume participation; 5bp per-side slippage',
        'fees': {'buy': .001, 'sell': .003},
        'action': ACTION,
        'fractional_entitlements': 'no cash or NAV credit; conservative omission pending verified settlement',
        'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57, 'alpha': .05},
        'family': 'all 18 neural paths x four policies x three order budgets x cash/nonimage comparators = 432',
        'gate': 'positive net return; at least two positive quarters; MDD above -20%; fewer than four low-turnover weeks; both comparators and blocks adjusted p<.05 and simultaneous lower>0',
        'limitations': [
            'Reused development observations and adaptive prior probes; no independent confirmation.',
            'Order count is a capacity sensitivity, not a reconstruction of historical order-book cooldown.',
            'Only the stated corporate action has an exact announcement-based override; other inferred events remain approximate.',
            'Model labels/features were trained on the frozen input data; this changes the execution ledger only.',
            'No minimum equity allocation is stated in official manual 104 read on 2026-09-28; weekly turnover remains mandatory.',
        ],
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Execution amendment changed; use a new version')
    if not path.exists():
        atomic_json(path, spec)
        frozen = dest / 'frozen_source'; frozen.mkdir(exist_ok=True)
        for p in files: shutil.copy2(p, frozen / p.name)
    return spec


def prediction_receipt(model_root, dest):
    paths = sorted((model_root / 'models').glob('*/*.pred.npy'))
    if len(paths) != 198: raise RuntimeError(f'Expected 198 monthly forecasts, found {len(paths)}')
    result = {str(p.relative_to(model_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    path = dest / 'forecast_hashes.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError('Frozen model forecasts changed')
    if not path.exists(): atomic_json(path, result)


def run(dest=DEST, model_root=MODEL_ROOT, source=SOURCE):
    dest, model_root = Path(dest), Path(model_root)
    spec = freeze(dest, model_root); model_spec = spec['model_protocol']
    prediction_receipt(model_root, dest)
    p, ix, ci, di = context(source); p, release = corrected_panel(p, ix)
    schedules, _ = market_regimes(p)
    scores, choices, neural = load_scores(model_root, p, ix, ci, di, model_spec)
    atomic_json(dest / 'online_choices.json', choices)
    folder = dest / 'portfolios'; folder.mkdir(exist_ok=True)
    rows, returns, audits = [], {}, {}
    for policy in model_spec['policies']:
        panel = dict(p); panel['sector_cap'] = np.minimum(p['sector_cap'], policy['sector_cap'])
        schedule = schedules[policy['regime']]
        for budget in spec['order_budgets']:
            for name, matrix in scores.items():
                key = f"{name}__{policy['id']}__orders{budget}"
                path = folder / (key + '.json')
                if path.exists(): result = json.loads(path.read_text())
                else:
                    result = replay(panel, ix, matrix, model_spec['evaluation_start'], model_spec['evaluation_end'],
                                    top_n=policy['top_n'], weight=policy['stock_weight'], gross=.8,
                                    gross_schedule=schedule, max_orders=budget, return_trades=True,
                                    planning_price='open', action_release_dates=release)
                    atomic_json(path, result)
                check = audit_fills(panel, ix, result)
                check.update(additional_checks(panel, ix, result, schedule, max_orders=budget))
                check['announced_action_errors'] = announced_action_check(panel, ix, result)
                if (check['post_buy_limit_violations'] or check['additional_errors'] or
                    check['announced_action_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01):
                    atomic_json(dest / 'failed_audit.json', {'id': key, 'audit': check})
                    raise AssertionError('Amended ledger audit failed')
                audits[key] = check
                nav = np.array([r['nav'] for r in result['daily']])
                returns[key] = nav / np.r_[INITIAL_NAV, nav[:-1]] - 1
                quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=name, policy=policy['id'], order_budget=budget,
                                 **result['metrics'], **quarters,
                                 positive_blocks=sum(v > 0 for v in quarters.values())))
            pd.DataFrame(rows).to_csv(dest / 'portfolio_summary.csv', index=False)
            atomic_json(dest / 'audit_progress.json', {'portfolios': len(audits), 'all_passed': True})
            print(json.dumps({'policy': policy['id'], 'orders': budget, 'portfolios': len(rows)}), flush=True)
    audit_summary = {'portfolios': len(audits), 'buy_checks': sum(x['buy_checks'] for x in audits.values()),
                     'post_buy_violations': 0, 'additional_errors': 0, 'announced_action_errors': 0,
                     'max_nav_error_krw': max(x['maximum_nav_reconstruction_error_krw'] for x in audits.values())}
    atomic_json(dest / 'independent_audit.json', {'summary': audit_summary, 'portfolios': audits})
    family, differences = [], []
    for policy in model_spec['policies']:
        for budget in spec['order_budgets']:
            suffix = f"__{policy['id']}__orders{budget}"
            control = returns['online_nonimage' + suffix]
            for name in neural:
                key = name + suffix
                for comparator, base in [('cash', 0.), ('nonimage', control)]:
                    family.append((key, comparator)); differences.append(returns[key] - base)
    if len(family) != 432: raise AssertionError('Hypothesis family differs from protocol')
    matrix = np.column_stack(differences); statistics = []
    for block in spec['bootstrap']['blocks']:
        result = family_bootstrap(matrix, block=block, draws=spec['bootstrap']['draws'], seed=spec['bootstrap']['seed'])
        for i, (key, comparator) in enumerate(family):
            statistics.append(dict(id=key, comparator=comparator, block=block,
                                   **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(dest / 'bootstrap.csv', index=False)
    candidates = []
    for row in rows:
        if row['model'] not in neural: continue
        inference = stats[stats.id == row['id']]
        gate = row['return'] > 0 and row['positive_blocks'] >= 2 and row['mdd'] > -.20 and not row['four_week_turnover_stop']
        if gate and (inference.adjusted_p < .05).all() and (inference.simultaneous_lower95 > 0).all():
            candidates.append(row['id'])
    frame = pd.DataFrame(rows)
    summary = {'status': 'development_only', 'portfolios': len(rows), 'family_hypotheses': len(family),
               'days': len(matrix), 'candidate_gate_passed': candidates, 'independent_confirmation': False,
               'top_return': frame.sort_values('return', ascending=False).head(10).replace({np.nan: None}).to_dict('records'),
               'audit': audit_summary}
    atomic_json(dest / 'evaluation_summary.json', summary)
    print(json.dumps({'complete': True, 'candidates': candidates, 'audit': audit_summary}), flush=True)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run'])
    args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

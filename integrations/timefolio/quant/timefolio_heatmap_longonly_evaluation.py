"""Matched after-cost targets with a predeclared optional cash gate.

All prior development hypotheses remain in the joint max-t family. Binary
scores are uncalibrated success scores, not expected returns or guarantees.
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
from quant.timefolio_heatmap_action_amendment import DEST as BASE_ROOT, amend_actions, release_audit
from quant.timefolio_heatmap_seed_evaluation import DEST as SEED_ROOT, daily_returns, merge_months
from quant.timefolio_heatmap_portfolio_selector import DEST as SELECTOR_ROOT
from quant.timefolio_heatmap_allocation_probe import DEST as ALLOCATION_ROOT
from quant.timefolio_heatmap_allocation_expansion import DEST as PRIOR_ROOT
from quant.timefolio_heatmap_longonly_training import DEST as TRAIN_ROOT, CONFIGS
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import SOURCE
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260928_longonly_evaluation_v1')
CASES = [dict(id=f'n{n}__orders{orders}__gate{int(gate)}', top_n=n, max_orders=orders, gate=gate)
         for n in [4, 12] for orders in [3, 10] for gate in [False, True]]


def gated_scores(raw, eligible, mode, enabled):
    """Missing forecasts hold; observed all-negative forecasts request cash."""
    raw = np.asarray(raw, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if raw.ndim != 2 or raw.shape != eligible.shape:
        raise ValueError('Scores and eligibility must have the same 2D shape')
    if mode not in ['binary', 'return']:
        raise ValueError('Unknown long-only target')
    scores = raw.copy(); schedule = np.full(raw.shape[1], .8)
    if enabled:
        valid = np.isfinite(raw) & eligible
        positive = valid & (raw > (.5 if mode == 'binary' else 0.))
        scores[~positive] = np.nan
        schedule[valid.any(0) & ~positive.any(0)] = 0.
    return scores, schedule


def freeze():
    training = json.loads((TRAIN_ROOT/'protocol.json').read_text())
    base = json.loads((BASE_ROOT/'protocol.json').read_text())
    files = [Path(__file__), TRAIN_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json', BASE_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['longonly_training', 'action_amendment', 'execution_amendment', 'planned_replay',
               'planned_audit', 'walkforward_eval', 'walkforward', 'study', 'features', 'replay',
               'data', 'seed_evaluation']]
    spec = {'training': training, 'base': base, 'cases': CASES,
            'policy': 'stock5%, sector min(statutory,20%), gross ceiling80%, rebalance5; no band; same gross schedule machinery in both arms',
            'gate': 'binary score strictly >.5 or scaled Huber net-return forecast >0; all observed eligible forecasts nonpositive requests gross0; missing all forecasts never requests cash',
            'gate_limits': 'binary scores are uncalibrated; >.5 success probability does not imply positive expected value; Huber outputs approximate a robust location, not guaranteed mean returns',
            'comparators': 'cash and same-label GBM under the identical allocation and gate; no model chosen using these results',
            'joint_family': '1032 prior plus two CNN targets x eight cases x two comparators =1064',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57},
            'status': 'adaptive development only; all prior data and execution limitations retained',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Long-only evaluation protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files): shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    return spec


def prior_family():
    prior = pd.read_csv(PRIOR_ROOT/'joint_bootstrap.csv'); family = []; differences = []; cache = {}
    roots = {'original': BASE_ROOT, 'replication': SEED_ROOT, 'portfolio_selector': SELECTOR_ROOT,
             'allocation': ALLOCATION_ROOT, 'allocation_expanded': PRIOR_ROOT}
    dates = None
    def returns(root, key):
        nonlocal dates
        name = (str(root), key)
        if name not in cache:
            result = json.loads((root/'portfolios'/(key+'.json')).read_text())
            current_dates = [r['date'] for r in result['daily']]
            if dates is None: dates = current_dates
            if current_dates != dates: raise AssertionError('Portfolio dates differ')
            cache[name] = daily_returns(result)
        return cache[name]
    for key, comp, origin in prior[['id', 'comparator', 'origin']].drop_duplicates().itertuples(index=False, name=None):
        source = roots[origin]
        if origin == 'portfolio_selector': control_root = source; control = key.replace('_neural__', '_nonimage__')
        elif origin in ['allocation', 'allocation_expanded']: control_root = source; control = 'online_nonimage__'+key.split('__', 1)[1]
        else: control_root = BASE_ROOT; control = 'online_nonimage__'+key.split('__', 1)[1]
        baseline = 0. if comp == 'cash' else returns(control_root, control)
        family.append((key, comp, origin)); differences.append(returns(source, key)-baseline)
    if len(family) != 1032: raise AssertionError('Prior hypothesis family incomplete')
    return family, differences, returns


def run():
    spec = freeze(); base = spec['base']; model_spec = base['prior']['model_protocol']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, base['actions'])
    p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    files = list((TRAIN_ROOT/'models').glob('*/*.pred.npy'))
    if len(files) != 36: raise AssertionError('Training is incomplete')
    atomic_json(DEST/'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True); rows = []; audits = {}
    for cfg in CONFIGS:
        values = merge_months(TRAIN_ROOT/'models'/cfg['id'], di, ix['dates'])
        raw = score_matrix(values, ci, di, p['close'].shape)
        for case in CASES:
            key = cfg['id']+'__'+case['id']; path = folder/(key+'.json')
            scores, schedule = gated_scores(raw, p['eligible'], cfg['mode'], case['gate'])
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(p, ix, scores, model_spec['evaluation_start'], model_spec['evaluation_end'],
                                top_n=case['top_n'], weight=.05, max_orders=case['max_orders'],
                                gross_schedule=schedule, return_trades=True, planning_price='open', action_release_dates=release)
                atomic_json(path, result)
            check = audit_fills(p, ix, result)
            check.update(additional_checks(p, ix, result, schedule, max_orders=case['max_orders']))
            check['announced_action_errors'] = release_audit(p, ix, result, release)
            if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['maximum_nav_reconstruction_error_krw'] > .01):
                atomic_json(DEST/'failed_audit.json', {'id': key, 'audit': check}); raise AssertionError('Long-only audit failed')
            audits[key] = check; quarters = calendar_blocks(result['daily'])
            rows.append(dict(id=key, model=cfg['id'], architecture=cfg['architecture'], mode=cfg['mode'],
                             case=case['id'], gate=case['gate'], top_n=case['top_n'], order_budget=case['max_orders'],
                             zero_target_signal_dates=int(np.sum(schedule == 0)), **result['metrics'],
                             **quarters, positive_blocks=sum(x > 0 for x in quarters.values())))
        print(json.dumps({'model': cfg['id'], 'portfolios': len(rows)}), flush=True)
    atomic_json(DEST/'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    family, differences, returns = prior_family()
    for row in frame[frame.architecture == 'cnn'].to_dict('records'):
        control = 'gbm_net_'+row['mode']+'_h5__'+row['case']
        for comp, baseline in [('cash', 0.), ('nonimage', returns(DEST, control))]:
            family.append((row['id'], comp, 'longonly')); differences.append(returns(DEST, row['id'])-baseline)
    if len(family) != 1064: raise AssertionError('Joint family changed')
    statistics = []
    for block in spec['bootstrap']['blocks']:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            statistics.append(dict(id=key, comparator=comp, origin=origin, block=block,
                                   **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in frame[frame.architecture == 'cnn'].to_dict('records'):
        infer = stats[(stats.id == row['id']) & (stats.origin == 'longonly')]
        if len(infer) != 4: raise AssertionError('Incomplete candidate comparisons')
        gate = row['return'] > 0 and row['mdd'] > -.20 and row['positive_blocks'] >= 2 and not row['four_week_turnover_stop']
        if gate and (infer.adjusted_p < .05).all() and (infer.simultaneous_lower95 > 0).all(): candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json', {'status': 'development_only', 'portfolios': len(rows),
                                               'joint_hypotheses': len(family), 'candidate_gate_passed': candidates,
                                               'independent_confirmation': False})
    print(json.dumps({'complete': True, 'candidates': candidates}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

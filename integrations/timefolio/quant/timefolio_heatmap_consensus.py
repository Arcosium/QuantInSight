"""Fixed cross-model rank consensus, with label-free and causal aggregation.

Compare neural and nonimage pools using the same mean, median and equal-family
rules. Portfolio outcomes never choose member weights or remove weak members.
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
from quant.timefolio_heatmap_execution_amendment import prediction_receipt
from quant.timefolio_heatmap_snapshot import DEST as PRIOR_ROOT, CASES as SNAPSHOT_CASES, completed_family, snapshot_scores
from quant.timefolio_heatmap_banded_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE, configurations
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_consensus_v1')
METHODS = ['mean', 'median', 'balanced']
CASES = [dict(c) for c in SNAPSHOT_CASES] + [
    dict(id=f'daily5__band{bp}bp__n{n}__orders{orders}', mode='daily5', rebalance=5,
         refresh=1, top_n=n, max_orders=orders, rebalance_band=bp/10000)
    for bp in [0, 5] for n in [12, 20] for orders in [3, 10]]


def rank_consensus(matrices, groups, eligible, method):
    """Ranks use only the current date; shared coverage prevents member drift."""
    if method not in METHODS or not matrices: raise ValueError('Unknown or empty consensus')
    eligible = np.asarray(eligible, bool); names = sorted(matrices); ranked = []
    support = None
    for name in names:
        a = np.asarray(matrices[name])
        if a.ndim != 2 or a.shape != eligible.shape or name not in groups:
            raise ValueError('Member shapes/groups differ')
        finite = np.isfinite(a) & eligible
        if support is None: support = finite
        elif not np.array_equal(support, finite): raise ValueError('Member coverage differs')
        ranked.append(pd.DataFrame(np.where(finite, a, np.nan)).rank(axis=0, pct=True).to_numpy(np.float32))
    a = np.stack(ranked)
    if method == 'mean': result = np.mean(a, axis=0)
    elif method == 'median': result = np.median(a, axis=0)
    else:
        families = sorted({groups[name] for name in names})
        result = np.mean([a[[groups[name] == family for name in names]].mean(0) for family in families], axis=0)
    return result.astype(np.float32)


def freeze():
    base = json.loads((BASE_ROOT/'protocol.json').read_text())
    pools = {'neural': {c['id']: c['architecture'] for c in configurations() if c['architecture'] in ['cnn', 'split', 'tcn']},
             'nonimage': {c['id']: c['architecture'] for c in configurations() if c['architecture'] in ['mlp', 'gbm']}}
    if len(pools['neural']) != 16 or len(pools['nonimage']) != 6: raise AssertionError('Member pool changed')
    files = [Path(__file__), BASE_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['snapshot', 'action_amendment', 'execution_amendment', 'banded_replay', 'planned_audit',
               'longonly_evaluation', 'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = {'base': base, 'pools': pools, 'methods': METHODS, 'cases': CASES,
            'aggregation': 'same-date percentile ranks among eligible shared observations; equal-member mean, member median, or equal-architecture average of within-architecture means',
            'balanced_weights': 'neural CNN/split/TCN each1/3; controls MLP/GBM each1/2; all original members retained; no outcome-dependent weighting',
            'execution': 'same snapshot/band grid plus daily5 reference grid; stock5%, sector min(statutory,20%), gross ceiling80%; frozen banded ledger and verified action treatment',
            'empty_session': 'preserve established no-signal behavior where current eligibility is entirely missing',
            'comparators': 'cash and nonimage consensus of the same method and portfolio case',
            'joint_family': '1928 prior plus three neural consensus methods x 32 cases x two comparators =2120',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57},
            'status': 'adaptive development; no new independent holdout or automatic trading promotion',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Consensus protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files): shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    prediction_receipt(MODEL_ROOT, DEST)
    if json.loads((DEST/'forecast_hashes.json').read_text()) != json.loads((BASE_ROOT/'forecast_hashes.json').read_text()):
        raise AssertionError('Original forecasts changed')
    return spec


def prior_comparisons():
    family, differences, returns, _ = completed_family()
    prior = pd.read_csv(PRIOR_ROOT/'joint_bootstrap.csv')
    added = prior[prior.origin == 'snapshot'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        control = 'online_nonimage__'+key.split('__', 1)[1]
        baseline = 0. if comp == 'cash' else returns(PRIOR_ROOT, control)
        family.append((key, comp, origin)); differences.append(returns(PRIOR_ROOT, key)-baseline)
    if len(family) != 1928 or len(set(family)) != 1928: raise AssertionError('Prior family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(np.mean(x)-means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Prior returns changed')
    return family, differences, returns, error


def run():
    spec = freeze(); base = spec['base']; model_spec = base['prior']['model_protocol']
    start, end = model_spec['evaluation_start'], model_spec['evaluation_end']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, base['actions'])
    p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    scores, _, _ = load_scores(MODEL_ROOT, p, ix, ci, di, model_spec)
    dates = ix['dates']; date_index = {date: d for d, date in enumerate(dates)}
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True); rows = []; audits = {}
    for pool, groups in spec['pools'].items():
        for method in METHODS:
            name = pool+'_'+method; matrix = rank_consensus({n: scores[n] for n in groups}, groups, p['eligible'], method)
            path = DEST/(name+'.npy')
            if path.exists():
                if not np.array_equal(np.load(path), matrix, equal_nan=True): raise AssertionError('Consensus changed')
            else: np.save(path, matrix)
            held = {interval: snapshot_scores(matrix, p['eligible'], dates, start, end, interval) for interval in [1, 5, 10]}
            for case in CASES:
                alpha, origins = held[case['refresh']]
                selected = np.flatnonzero((origins >= 0) & p['eligible'].any(0))
                if not np.all((np.isfinite(alpha[:, selected]) & p['eligible'][:, selected]).any(0)):
                    raise AssertionError('Missing eligible consensus')
                key = name+'__'+case['id']; path = folder/(key+'.json')
                if path.exists(): result = json.loads(path.read_text())
                else:
                    result = replay(p, ix, alpha, start, end, top_n=case['top_n'], weight=.05,
                                    rebalance=case['rebalance'], max_orders=case['max_orders'],
                                    rebalance_band=case['rebalance_band'], return_trades=True,
                                    planning_price='open', action_release_dates=release)
                    for trade in result['trades']:
                        signal = date_index[trade['signal_date']]; origin = origins[signal]
                        if not 0 <= origin <= signal: raise AssertionError('Future consensus')
                        trade['portfolio_score_date'] = dates[origin]
                    atomic_json(path, result)
                check = audit_fills(p, ix, result)
                check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
                check['announced_action_errors'] = release_audit(p, ix, result, release)
                check['snapshot_origin_errors'] = []
                for trade in result['trades']:
                    d = date_index[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d or trade['portfolio_score_date'] != dates[origin]:
                        check['snapshot_origin_errors'].append(trade['date'])
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                        or check['snapshot_origin_errors'] or check['maximum_nav_reconstruction_error_krw'] > .01):
                    atomic_json(DEST/'failed_audit.json', {'id': key, 'audit': check}); raise AssertionError('Consensus audit failed')
                audits[key] = check; quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, model=name, pool=pool, method=method, case=case['id'], mode=case['mode'],
                                 top_n=case['top_n'], order_budget=case['max_orders'], band=case['rebalance_band'],
                                 **result['metrics'], **quarters, positive_blocks=sum(x > 0 for x in quarters.values())))
            print(json.dumps({'consensus': name, 'portfolios': len(rows)}), flush=True)
    atomic_json(DEST/'independent_audit.json', audits); frame = pd.DataFrame(rows)
    frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_comparisons()
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses': len(family), 'mean_max_error': error})
    for row in frame[frame.pool == 'neural'].to_dict('records'):
        control = 'nonimage_'+row['method']+'__'+row['case']
        for comp, baseline in [('cash', 0.), ('nonimage', returns(DEST, control))]:
            family.append((row['id'], comp, 'consensus')); differences.append(returns(DEST, row['id'])-baseline)
    if len(family) != 2120: raise AssertionError('Joint family changed')
    stats = []
    for block in spec['bootstrap']['blocks']:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            stats.append(dict(id=key, comparator=comp, origin=origin, block=block,
                              **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    statistics = pd.DataFrame(stats); statistics.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in frame[frame.pool == 'neural'].to_dict('records'):
        infer = statistics[(statistics.id == row['id']) & (statistics.origin == 'consensus')]
        if len(infer) != 4: raise AssertionError('Missing candidate comparisons')
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

"""Higher target weights under dated stock limits and tight order budgets.

All original forecasts and fixed consensus pools are retained. The new weights
are 8%, 10% and 14%; 14% leaves distance from the general statutory 15% limit.
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
from quant.timefolio_heatmap_allocation_expansion import MODELS as ORIGINAL_MODELS, NEURAL as ORIGINAL_NEURAL
from quant.timefolio_heatmap_consensus import DEST as PRIOR_ROOT, METHODS, prior_comparisons
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import DEST as LIMIT_ROOT, historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import DEST as WEEK_ROOT, assess_weeks
from quant.timefolio_heatmap_dated_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_concentration_v1')
MODELS = ORIGINAL_MODELS+[pool+'_'+method for pool in ['neural', 'nonimage'] for method in METHODS]
NEURAL = ORIGINAL_NEURAL+['neural_'+method for method in METHODS]
CASES = [dict(id=f'w{bp}bp__n{n}__orders{orders}__{mode}', weight=bp/10000,
              top_n=n, max_orders=orders, mode=mode, refresh=refresh)
         for bp in [800, 1000, 1400] for n in [4, 8] for orders in [3, 10]
         for mode, refresh in [('daily5', 1), ('held5', 5)]]


def comparator(model):
    return model.replace('neural_', 'nonimage_', 1) if model.startswith('neural_') else 'online_nonimage'


def freeze():
    base = json.loads((BASE_ROOT/'protocol.json').read_text())
    audit = json.loads((LIMIT_ROOT/'summary.json').read_text())
    if audit['post_buy_failed_paths'] or audit['closing_breach_paths']:
        raise AssertionError('Correct affected prior paths before retaining old hypotheses')
    if audit['portfolios'] != 1976: raise AssertionError('Prior position-limit audit is incomplete')
    weeks = json.loads((WEEK_ROOT/'summary.json').read_text())
    if weeks['portfolios'] != 1976 or weeks['previously_failed_becoming_pass']:
        raise AssertionError('Boundary-week correction needs review')
    files = [Path(__file__), BASE_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json', LIMIT_ROOT/'summary.json',
             LIMIT_ROOT/'protocol.json', WEEK_ROOT/'summary.json', WEEK_ROOT/'protocol.json', WEEK_ROOT/'krx_2026_closures.pdf']
    files += [PRIOR_ROOT/(pool+'_'+method+'.npy') for pool in ['neural', 'nonimage'] for method in METHODS]
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['stock_limits', 'week_boundaries', 'dated_replay', 'consensus', 'snapshot', 'action_amendment',
               'execution_amendment', 'planned_audit', 'walkforward_eval', 'walkforward', 'study',
               'features', 'replay', 'data']]
    spec = {'base': base, 'models': MODELS, 'neural': NEURAL, 'cases': CASES,
            'policy': '8/10/14% target, maximum4/8 names, 3/10 daily fills; rebalance5; current or five-session snapshot; NAV5bp adjustment band; sector min(statutory,20%), small-cap aggregate30%, gross ceiling80%',
            'stock_limits': 'execution-date Hynix15% before20260701 and30% thereafter; Samsung40%; others15%; planning, repairs and fee-aware headroom use the dated vector',
            'prior_audit': audit,
            'week_boundaries': weeks,
            'turnover': 'include completed last week20260921..23 using verified KRX closures on24/25; partial first week remains excluded; returns unaffected',
            'comparators': 'cash and same-case online_nonimage for original models; corresponding fixed nonimage consensus for fixed neural consensus',
            'joint_family': '2120 prior plus 21 neural score paths x24 cases x2 comparators =3128',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57},
            'interpretation': 'adaptive development on reused dates; concentration raises risk and is not a leverage or live-order recommendation; prior corporate/book/universe limitations remain',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Concentration protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix in ['.py', '.json']: shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    prediction_receipt(MODEL_ROOT, DEST)
    if json.loads((DEST/'forecast_hashes.json').read_text()) != json.loads((BASE_ROOT/'forecast_hashes.json').read_text()):
        raise AssertionError('Original forecasts changed')
    return spec


def prior_family():
    family, differences, returns, _ = prior_comparisons()
    prior = pd.read_csv(PRIOR_ROOT/'joint_bootstrap.csv')
    added = prior[prior.origin == 'consensus'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        control = key.replace('neural_', 'nonimage_', 1)
        baseline = 0. if comp == 'cash' else returns(PRIOR_ROOT, control)
        family.append((key, comp, origin)); differences.append(returns(PRIOR_ROOT, key)-baseline)
    if len(family) != 2120 or len(set(family)) != 2120: raise AssertionError('Prior family incomplete')
    means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    error = max(abs(np.mean(x)-means.loc[key]) for key, x in zip(family, differences))
    if error > 1e-14: raise AssertionError('Prior returns changed')
    return family, differences, returns, error


def run():
    spec = freeze(); base = spec['base']; model_spec = base['prior']['model_protocol']
    start, end = model_spec['evaluation_start'], model_spec['evaluation_end']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, base['actions'])
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    scores, _, _ = load_scores(MODEL_ROOT, p, ix, ci, di, model_spec)
    for pool in ['neural', 'nonimage']:
        for method in METHODS: scores[pool+'_'+method] = np.load(PRIOR_ROOT/(pool+'_'+method+'.npy'))
    dates = ix['dates']; date_index = {date: d for d, date in enumerate(dates)}
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True); rows = []; audits = {}; regressions = []
    for name in MODELS:
        # The known correction must not alter previous 5% baseline accounts.
        daily, _ = snapshot_scores(scores[name], p['eligible'], dates, start, end, 1)
        check = replay(p, ix, daily, start, end, top_n=20, weight=.05, max_orders=3,
                       return_trades=True, planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
        if name in ORIGINAL_MODELS: original_path = BASE_ROOT/'portfolios'/f'{name}__sector20__orders3.json'
        else: original_path = PRIOR_ROOT/'portfolios'/f'{name}__daily5__band0bp__n20__orders3.json'
        original = json.loads(original_path.read_text())
        for trade in original['trades']: trade.pop('portfolio_score_date', None)
        check['metrics'].pop('execution_date_stock_limits')
        if check != original: raise AssertionError('Dated-limit baseline changed; recheck earlier results first')
        regressions.append({'model': name, 'all_baseline_fields_equal': True})
        held = {interval: snapshot_scores(scores[name], p['eligible'], dates, start, end, interval) for interval in [1, 5]}
        for case in CASES:
            alpha, origins = held[case['refresh']]
            selected = np.flatnonzero((origins >= 0) & p['eligible'].any(0))
            if not np.all((np.isfinite(alpha[:, selected]) & p['eligible'][:, selected]).any(0)):
                raise AssertionError('Missing eligible scores')
            key = name+'__'+case['id']; path = folder/(key+'.json')
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(p, ix, alpha, start, end, top_n=case['top_n'], weight=case['weight'],
                                rebalance=5, max_orders=case['max_orders'], rebalance_band=.0005,
                                return_trades=True, planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
                for trade in result['trades']:
                    d = date_index[trade['signal_date']]; origin = origins[d]
                    if not 0 <= origin <= d: raise AssertionError('Future score')
                    trade['portfolio_score_date'] = dates[origin]
                result['metrics'].update(assess_weeks(result))
                atomic_json(path, result)
            check = audit_fills(p, ix, result)
            check.update(additional_checks(p, ix, result, None, max_orders=case['max_orders']))
            check['announced_action_errors'] = release_audit(p, ix, result, release)
            check['dated_hynix_audit'] = audit_pre_july_hynix(p, ix, result)
            check['snapshot_origin_errors'] = []
            for trade in result['trades']:
                d = date_index[trade['signal_date']]; origin = origins[d]
                if not 0 <= origin <= d or trade['portfolio_score_date'] != dates[origin]: check['snapshot_origin_errors'].append(trade['date'])
            if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['snapshot_origin_errors'] or check['dated_hynix_audit']['post_buy_limit_errors']
                    or check['maximum_nav_reconstruction_error_krw'] > .01):
                atomic_json(DEST/'failed_audit.json', {'id': key, 'audit': check}); raise AssertionError('Concentration audit failed')
            audits[key] = check; quarters = calendar_blocks(result['daily'])
            rows.append(dict(id=key, model=name, case=case['id'], mode=case['mode'], top_n=case['top_n'],
                             weight=case['weight'], order_budget=case['max_orders'], **result['metrics'],
                             **quarters, positive_blocks=sum(x > 0 for x in quarters.values())))
        print(json.dumps({'model': name, 'portfolios': len(rows)}), flush=True)
    atomic_json(DEST/'baseline_regression.json', regressions); atomic_json(DEST/'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    family, differences, returns, error = prior_family()
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses': len(family), 'mean_max_error': error})
    for row in frame[frame.model.isin(NEURAL)].to_dict('records'):
        control = comparator(row['model'])+'__'+row['case']
        for comp, baseline in [('cash', 0.), ('nonimage', returns(DEST, control))]:
            family.append((row['id'], comp, 'concentration')); differences.append(returns(DEST, row['id'])-baseline)
    if len(family) != 3128: raise AssertionError('Joint family changed')
    stats = []
    for block in spec['bootstrap']['blocks']:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            stats.append(dict(id=key, comparator=comp, origin=origin, block=block,
                              **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    statistics = pd.DataFrame(stats); statistics.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in frame[frame.model.isin(NEURAL)].to_dict('records'):
        infer = statistics[(statistics.id == row['id']) & (statistics.origin == 'concentration')]
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

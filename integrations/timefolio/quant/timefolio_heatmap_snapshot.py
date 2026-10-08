"""Hold alpha snapshots between scheduled rebalances under order budgets.

Only score refresh changes. The frozen ledger still rechecks current admission,
sector/stock/size limits, opening plan prices and execution constraints. Target
quantities are recalculated; this is not a historical resting-order simulation.
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
from quant.timefolio_heatmap_allocation_expansion import DEST as ALLOCATION_ROOT, MODELS, NEURAL
from quant.timefolio_heatmap_longonly_evaluation import DEST as PRIOR_ROOT, prior_family
from quant.timefolio_heatmap_banded_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_snapshot_v2')
CASES = [dict(id=f'{mode}__band{bp}bp__n{n}__orders{orders}', mode=mode, rebalance=rebalance,
              refresh=refresh, top_n=n, max_orders=orders, rebalance_band=bp/10000)
         for mode, rebalance, refresh in [('held5', 5, 5), ('daily10', 10, 1), ('held10', 10, 10)]
         for bp in [0, 5] for n in [12, 20] for orders in [3, 10]]


def snapshot_scores(raw, eligible, dates, start, end, interval):
    """Each execution cycle starts with the preceding close's available scores."""
    raw = np.asarray(raw); eligible = np.asarray(eligible, bool)
    if raw.ndim != 2 or raw.shape != eligible.shape or raw.shape[1] != len(dates):
        raise ValueError('Score, eligibility and date shapes differ')
    if not isinstance(interval, int) or interval < 1:
        raise ValueError('Snapshot interval must be a positive integer')
    if list(dates) != sorted(set(dates)):
        raise ValueError('Dates must be strictly ordered')
    out = np.full(raw.shape, np.nan, dtype=np.float32); origins = np.full(len(dates), -1, int)
    days = [d for d, date in enumerate(dates) if start <= date <= end and d > 0]
    for k, d in enumerate(days):
        if k % interval == 0: origin = d-1
        out[:, d-1] = np.where(eligible[:, origin], raw[:, origin], np.nan)
        origins[d-1] = origin
    return out, origins


def freeze():
    base = json.loads((BASE_ROOT/'protocol.json').read_text())
    files = [Path(__file__), BASE_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json', ALLOCATION_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['action_amendment', 'execution_amendment', 'planned_replay', 'banded_replay', 'planned_audit',
               'longonly_evaluation', 'allocation_expansion', 'walkforward_eval', 'walkforward',
               'study', 'features', 'replay', 'data']]
    spec = {'base': base, 'models': MODELS, 'neural': NEURAL, 'cases': CASES,
            'prefill_amendment': 'v1 stopped before any portfolio result: 20260102 has zero eligible securities in the existing panel. Preserve original no-signal ledger behavior on zero-eligibility days; no target refresh or forced liquidation. Snapshot grid itself is unchanged.',
            'policy': 'stock5%, sector min(statutory,20%), gross ceiling80%, no exposure regime; continuing-position adjustment band 0 or NAV5bp',
            'snapshot': 'fixed scores and score-origin eligibility at each 5/10 execution-session boundary; first execution uses preceding-close snapshot; current eligibility and rules still checked daily',
            'execution': 'existing banded planned replay; retries can recompute quantities from current prices and NAV, but use held scores; not fixed quantity or resting orders; full exits/new entries/known risk repairs bypass band',
            'comparators': 'cash and online_nonimage under the same refresh/rebalance/allocation; original daily-score rebalance5 paths retained as descriptive baselines',
            'diagnostic': 'deferred requests and sells with a recent buy motivate the axis; recent-buy counts include position adjustments, not necessarily round-trip exits',
            'joint_family': '1064 prior hypotheses plus 18 neural score paths x 24 cases x two comparators =1928',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 57},
            'status': 'adaptive development; unchanged prior data and execution limitations; fresh 20260928 outcomes remain reserved',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Snapshot protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files): shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    prediction_receipt(MODEL_ROOT, DEST)
    return spec


def baseline_path(model, n, orders):
    if n == 20: return BASE_ROOT/'portfolios'/f'{model}__sector20__orders{orders}.json'
    if n == 12: return ALLOCATION_ROOT/'portfolios'/f'{model}__n12__orders{orders}__band0bp.json'
    raise ValueError('No matching frozen baseline')


def completed_family():
    family, differences, returns = prior_family()
    prior = pd.read_csv(PRIOR_ROOT/'joint_bootstrap.csv')
    added = prior[prior.origin == 'longonly'][['id', 'comparator', 'origin']].drop_duplicates()
    for key, comp, origin in added.itertuples(index=False, name=None):
        control = key.replace('cnn_net_', 'gbm_net_', 1)
        baseline = 0. if comp == 'cash' else returns(PRIOR_ROOT, control)
        family.append((key, comp, origin)); differences.append(returns(PRIOR_ROOT, key)-baseline)
    if len(family) != 1064 or len(set(family)) != 1064: raise AssertionError('Prior family incomplete')
    old_means = prior[prior.block == 5].set_index(['id', 'comparator', 'origin'])['mean']
    errors = [abs(np.mean(x)-old_means.loc[key]) for key, x in zip(family, differences)]
    if max(errors) > 1e-14: raise AssertionError('Prior returns changed')
    return family, differences, returns, max(errors)


def run():
    spec = freeze(); base = spec['base']; model_spec = base['prior']['model_protocol']
    start, end = model_spec['evaluation_start'], model_spec['evaluation_end']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, base['actions'])
    p['sector_cap'] = np.minimum(p['sector_cap'], .20)
    scores, _, _ = load_scores(MODEL_ROOT, p, ix, ci, di, model_spec)
    dates = ix['dates']; date_index = {date: d for d, date in enumerate(dates)}
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True); rows = []; audits = {}; regressions = []
    for name in MODELS:
        daily, _ = snapshot_scores(scores[name], p['eligible'], dates, start, end, 1)
        check = replay(p, ix, daily, start, end, top_n=20, weight=.05, max_orders=3,
                       return_trades=True, planning_price='open', action_release_dates=release)
        original = json.loads(baseline_path(name, 20, 3).read_text())
        if check != original: raise AssertionError('Daily refresh changed the frozen ledger')
        regressions.append({'model': name, 'case': 'sector20__orders3', 'all_fields_equal': True})
        held = {interval: snapshot_scores(scores[name], p['eligible'], dates, start, end, interval) for interval in [1, 5, 10]}
        for case in CASES:
            matrix, origins = held[case['refresh']]
            selected = np.flatnonzero((origins >= 0) & p['eligible'].any(0))
            if not np.all(np.isfinite(matrix[:, selected]).any(0)):
                raise AssertionError('A scheduled snapshot is entirely missing')
            if not np.all((np.isfinite(matrix[:, selected]) & p['eligible'][:, selected]).any(0)):
                raise AssertionError('No currently eligible snapshot score; define this case before proceeding')
            key = name+'__'+case['id']; path = folder/(key+'.json')
            if path.exists(): result = json.loads(path.read_text())
            else:
                result = replay(p, ix, matrix, start, end, top_n=case['top_n'], weight=.05,
                                rebalance=case['rebalance'], max_orders=case['max_orders'],
                                rebalance_band=case['rebalance_band'],
                                return_trades=True, planning_price='open', action_release_dates=release)
                for trade in result['trades']:
                    signal = date_index[trade['signal_date']]; origin = origins[signal]
                    if not 0 <= origin <= signal: raise AssertionError('Future snapshot')
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
                atomic_json(DEST/'failed_audit.json', {'id': key, 'audit': check}); raise AssertionError('Snapshot audit failed')
            audits[key] = check; quarters = calendar_blocks(result['daily'])
            baseline = json.loads(baseline_path(name, case['top_n'], case['max_orders']).read_text())['metrics']
            row = dict(id=key, model=name, case=case['id'], mode=case['mode'], top_n=case['top_n'],
                       order_budget=case['max_orders'], rebalance=case['rebalance'], refresh=case['refresh'],
                       band=case['rebalance_band'],
                       **result['metrics'], **quarters, positive_blocks=sum(x > 0 for x in quarters.values()))
            row.update({f'{key}_change_vs_daily5': row[key]-baseline[key] for key in
                        ['return', 'fees_krw', 'mean_gross', 'weekly_turnover_mean', 'low_turnover_weeks']})
            rows.append(row)
        print(json.dumps({'model': name, 'portfolios': len(rows)}), flush=True)
    atomic_json(DEST/'daily_regression.json', regressions)
    atomic_json(DEST/'independent_audit.json', audits)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    family, differences, returns, error = completed_family()
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses': len(family), 'mean_max_error': error})
    for row in frame[frame.model.isin(NEURAL)].to_dict('records'):
        for comp, baseline in [('cash', 0.), ('nonimage', returns(DEST, 'online_nonimage__'+row['case']))]:
            family.append((row['id'], comp, 'snapshot')); differences.append(returns(DEST, row['id'])-baseline)
    if len(family) != 1928: raise AssertionError('Joint family changed')
    stats = []
    for block in spec['bootstrap']['blocks']:
        result = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=57)
        for i, (key, comp, origin) in enumerate(family):
            stats.append(dict(id=key, comparator=comp, origin=origin, block=block,
                              **{k: float(result[k][i]) for k in ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    statistics = pd.DataFrame(stats); statistics.to_csv(DEST/'joint_bootstrap.csv', index=False); candidates = []
    for row in frame[frame.model.isin(NEURAL)].to_dict('records'):
        infer = statistics[(statistics.id == row['id']) & (statistics.origin == 'snapshot')]
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

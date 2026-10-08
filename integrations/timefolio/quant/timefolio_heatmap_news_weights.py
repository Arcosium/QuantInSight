"""Incremental archived macro effect when it directly scales stock targets.

This is a paired overlay study on the same three causal selectors as the earlier
macro study. It cannot establish independent heatmap alpha on reused history.
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
from quant.timefolio_heatmap_action_amendment import DEST as ACTION_ROOT, amend_actions, release_audit
from quant.timefolio_heatmap_execution_amendment import prediction_receipt
from quant.timefolio_heatmap_macro_action_check import DEST as PRIOR_ROOT
from quant.timefolio_heatmap_macro_amendment import COMPARISONS as PRIOR_COMPARISONS
from quant.timefolio_heatmap_macro_overlay import MACRO_ROOT, SELECTORS, schedules_from_records
from quant.timefolio_heatmap_dated_replay import replay as prior_replay
from quant.timefolio_heatmap_weighted_replay import replay
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_replay import INITIAL_NAV
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_news_weights_v1')
CASES = [dict(id=f'w{bp}bp__orders{budget}', weight=bp/10000, max_orders=budget)
         for bp in [500, 1000] for budget in [3, 10]]
COMPARISONS = PRIOR_COMPARISONS + [('macro_stock', 'fixed80')]


def freeze():
    action = json.loads((ACTION_ROOT/'protocol.json').read_text())
    prior = json.loads((PRIOR_ROOT/'protocol.json').read_text())
    files = [Path(__file__), ACTION_ROOT/'protocol.json', PRIOR_ROOT/'protocol.json',
             PRIOR_ROOT/'bootstrap.csv', MACRO_ROOT/'daily_signals.json', MACRO_ROOT/'readiness.json']
    files += sorted((PRIOR_ROOT/'portfolios').glob('*.json'))
    files += [Path(__file__).with_name('timefolio_heatmap_'+name+'.py') for name in
              ['weighted_replay', 'dated_replay', 'stock_limits', 'week_boundaries', 'macro_action_check',
               'macro_amendment', 'macro_overlay', 'macro_archive', 'action_amendment',
               'execution_amendment', 'planned_audit', 'walkforward_eval', 'walkforward',
               'study', 'features', 'replay', 'data']]
    spec = {'start': '20260519', 'end': '20260923', 'selectors': SELECTORS, 'cases': CASES,
            'arms': ['fixed60', 'price_vol', 'macro_stock', 'price_macro_min', 'macro_smooth', 'fixed80'],
            'action': action, 'prior': prior,
            'weight': 'base stock target (5% or10%) times same archived gross recommendation /0.8; fixed80 factor1; fixed60 factor0.75; positive targets only',
            'availability': 'same first-available archived report, 15:30 signal cutoff, age<=96h; executed next session; stock recommendation clip30..80%, missing60%; prior20 smooth excludes current report',
            'policy': 'same score stream; top12, rebalance5 with unchanged repair/pending behavior, NAV5bp band, fixed gross ceiling80%; sector min(statutory,20%), small-cap30%, execution-date stock caps',
            'retraining': 'none; forecast hashes must match corrected action study',
            'comparisons': [list(x) for x in COMPARISONS],
            'family': '27 previous paired macro hypotheses retained plus 3selectors x4cases x4comparisons =75; same87 execution dates, same initially cash accounts',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 81},
            'gate': 'paired incremental news effect must pass both blocks; within75 max-t adjusted p<0.025 and simultaneous lower95>0. Threshold splits 0.05 between this macro family and the separate full-period heatmap family. No heatmap promotion from overlay evidence alone.',
            'interpretation': 'adaptive development on reused short history; mixed news/index/disclosure/prior-allocation signal, not pure news; prior corporate/book/universe limitations; fixed60 and past20 controls do not exactly match realized gross',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('News-weight protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    prediction_receipt(MODEL_ROOT, DEST)
    if json.loads((DEST/'forecast_hashes.json').read_text()) != json.loads((ACTION_ROOT/'forecast_hashes.json').read_text()):
        raise AssertionError('Forecasts changed')
    return spec


def checked(panel, ix, result, release, budget, schedule=None):
    check = audit_fills(panel, ix, result)
    check.update(additional_checks(panel, ix, result, schedule, max_orders=budget))
    check['announced_action_errors'] = release_audit(panel, ix, result, release)
    check['dated_hynix_audit'] = audit_pre_july_hynix(panel, ix, result)
    if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
            or check['dated_hynix_audit']['post_buy_limit_errors']
            or check['maximum_nav_reconstruction_error_krw'] > .01):
        raise AssertionError('News-weight ledger audit failed')
    return check


def run():
    spec = freeze(); action = spec['action']; start, end = spec['start'], spec['end']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, action['actions'])
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    schedules, available = schedules_from_records(p, ix['dates'], json.loads((MACRO_ROOT/'daily_signals.json').read_text()))
    scores, choices, _ = load_scores(MODEL_ROOT, p, ix, ci, di, action['prior']['model_protocol'])
    atomic_json(DEST/'online_choices.json', choices)
    dates = np.asarray(ix['dates']); used = np.flatnonzero((dates >= start) & (dates <= end))
    signals = used-1; expected_dates = dates[used].tolist()
    schedule_table = pd.DataFrame({'signal_date': dates[signals], 'execution_date': dates[used],
                                   'macro_available': available[signals]})
    for name, schedule in schedules.items(): schedule_table[name] = schedule[signals]/.8
    schedule_table['fixed80'] = 1.; schedule_table.to_csv(DEST/'target_factors.csv', index=False)
    returns = {}

    def series(result):
        if [r['date'] for r in result['daily']] != expected_dates: raise AssertionError('Unequal daily axes')
        nav = np.array([r['nav'] for r in result['daily']])
        return nav/np.r_[INITIAL_NAV, nav[:-1]]-1

    # Reconstruct every retained macro path under dated limits before reusing it.
    panel = dict(p); panel['sector_cap'] = np.minimum(p['sector_cap'], .15)
    regression = []; prior_audits = {}
    for budget in spec['prior']['parent']['order_budgets']:
        for selector in SELECTORS:
            for arm, schedule in schedules.items():
                key = f'{selector}__{arm}__orders{budget}'
                original = json.loads((PRIOR_ROOT/'portfolios'/(key+'.json')).read_text())
                check = replay(panel, ix, scores[selector], start, end, top_n=24, weight=.04,
                               gross_schedule=schedule, max_orders=budget, return_trades=True,
                               planning_price='open', action_release_dates=release, stock_cap_schedule=caps)
                check['metrics'].pop('execution_date_stock_limits')
                if check != original: raise AssertionError('Previous macro path changed')
                regression.append({'id': key, 'all_fields_equal': True, 'corrected_weeks': assess_weeks(check)})
                prior_audits[key] = checked(panel, ix, check, release, budget, schedule)
                returns[('prior', key)] = series(original)
    if len(regression) != 45: raise AssertionError('Incomplete macro reconstruction')
    atomic_json(DEST/'prior_replay_regression.json', regression)
    atomic_json(DEST/'prior_independent_audit.json', prior_audits)
    schedules['fixed80'] = np.full(len(dates), .8)
    panel = dict(p); panel['sector_cap'] = np.minimum(p['sector_cap'], .20)
    folder = DEST/'portfolios'; folder.mkdir(exist_ok=True)
    rows = []; audits = {}; results = {}; constant_regressions = []
    for selector in SELECTORS:
        for case in CASES:
            for arm, scale in schedules.items():
                key = selector+'__'+case['id']+'__'+arm; path = folder/(key+'.json')
                kw = dict(top_n=12, weight=case['weight'], rebalance=5, rebalance_band=.0005,
                          max_orders=case['max_orders'], return_trades=True, planning_price='open',
                          action_release_dates=release, stock_cap_schedule=caps)
                if path.exists(): result = json.loads(path.read_text())
                else:
                    result = replay(panel, ix, scores[selector], start, end,
                                    weight_schedule=case['weight']*(scale/.8), **kw)
                    result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                if arm == 'fixed80':
                    original = prior_replay(panel, ix, scores[selector], start, end, **kw)
                    original['metrics'].update(assess_weeks(original))
                    copied = dict(result); copied['metrics'] = dict(result['metrics'])
                    copied['metrics'].pop('mean_target_stock_weight')
                    if copied != original: raise AssertionError('Constant-weight replay changed')
                    constant_regressions.append({'id': key, 'all_fields_equal_except_schedule_metadata': True})
                audits[key] = checked(panel, ix, result, release, case['max_orders'])
                returns[('new', key)] = series(result); results[key] = result
                quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, selector=selector, case=case['id'], arm=arm,
                                 **result['metrics'], **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
            print(json.dumps({'selector': selector, 'case': case['id'], 'portfolios': len(rows)}), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv', index=False)
    atomic_json(DEST/'independent_audit.json', audits)
    atomic_json(DEST/'constant_regression.json', constant_regressions)
    family = []; differences = []; paired = []
    for budget in spec['prior']['parent']['order_budgets']:
        for selector in SELECTORS:
            for treatment, control in PRIOR_COMPARISONS:
                tk = f'{selector}__{treatment}__orders{budget}'; ck = f'{selector}__{control}__orders{budget}'
                family.append(dict(origin='prior', selector=selector, case=f'orders{budget}', treatment=treatment, control=control))
                differences.append(returns[('prior', tk)]-returns[('prior', ck)])
    prior_stats = pd.read_csv(PRIOR_ROOT/'bootstrap.csv'); old = prior_stats[prior_stats.block == 5]
    error = 0.
    for h, diff in zip(family, differences):
        row = old[(old.selector == h['selector']) & (old.order_budget == int(h['case'][6:]))
                  & (old.treatment == h['treatment']) & (old.control == h['control'])]
        if len(row) != 1: raise AssertionError('Missing retained comparison')
        error = max(error, abs(float(row.iloc[0]['mean'])-float(diff.mean())))
    if error > 1e-14: raise AssertionError('Prior news comparison changed')
    atomic_json(DEST/'prior_family_reconstruction.json', {'hypotheses': len(family), 'mean_max_error': error})
    for selector in SELECTORS:
        for case in CASES:
            for treatment, control in COMPARISONS:
                prefix = selector+'__'+case['id']+'__'; tk, ck = prefix+treatment, prefix+control
                h = dict(origin='new', selector=selector, case=case['id'], treatment=treatment, control=control)
                family.append(h); differences.append(returns[('new', tk)]-returns[('new', ck)])
                a, b = results[tk], results[ck]
                paired.append(dict(h, treatment_id=tk, control_id=ck,
                                   trades_changed=a['trades'] != b['trades'], daily_changed=a['daily'] != b['daily'],
                                   return_difference=a['metrics']['return']-b['metrics']['return'],
                                   gross_difference=a['metrics']['mean_gross']-b['metrics']['mean_gross'],
                                   treatment_turnover_fail=a['metrics']['four_week_turnover_stop']))
    if len(family) != 75: raise AssertionError('Macro family changed')
    pd.DataFrame(paired).to_csv(DEST/'paired_effects.csv', index=False)
    statistics = []
    for block in spec['bootstrap']['blocks']:
        boot = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=81)
        for i, h in enumerate(family):
            statistics.append(dict(h, block=block, **{k: float(boot[k][i]) for k in
                              ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    pd.DataFrame(statistics).to_csv(DEST/'joint_bootstrap.csv', index=False)
    passed = []
    for h in family:
        inference = [r for r in statistics if all(r[k] == v for k, v in h.items())]
        if len(inference) != 2: raise AssertionError('Incomplete paired block comparison')
        if all(r['adjusted_p'] < .025 and r['simultaneous_lower95'] > 0 for r in inference): passed.append(h)
    summary = {'status': 'development_only', 'portfolios': len(rows), 'retained_portfolios': len(regression),
               'execution_sessions': len(used), 'covered_signal_sessions': int(available[signals].sum()),
               'joint_incremental_news_hypotheses': len(family), 'passed_incremental_effects': passed,
               'new_pairs_with_changed_trades': sum(r['trades_changed'] for r in paired),
               'minimum_adjusted_p': min(r['adjusted_p'] for r in statistics),
               'heatmap_promotion_assessed': False, 'independent_confirmation': False}
    atomic_json(DEST/'evaluation_summary.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

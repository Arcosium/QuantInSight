"""Same archived macro signal, observed by execution08:55 instead of prior close."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_macro_archive import KST, timestamp, first_available
from quant.timefolio_heatmap_macro_overlay import MACRO_ROOT, SELECTORS, schedules_from_records
from quant.timefolio_heatmap_news_weights import DEST as PRIOR, PRIOR_ROOT as GROSS_ROOT, CASES, checked
from quant.timefolio_heatmap_action_amendment import amend_actions
from quant.timefolio_heatmap_weighted_replay import replay
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT
from quant.timefolio_heatmap_walkforward_eval import load_scores, calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_morning_news_v2')
READINESS = SOURCE.with_name('20260929_morning_macro_readiness_v1')
COMPARISONS = [('morning_stock', 'fixed60'), ('morning_stock', 'fixed80'),
               ('morning_stock', 'morning_smooth'), ('morning_stock', 'macro_stock'),
               ('morning_price_min', 'price_vol'), ('morning_price_min', 'price_macro_min')]


def morning_signals(records, dates):
    """Index by stock signal date, but use the next session's08:55 deadline.

    The final calendar date has no execution session in the supplied calendar;
    it receives an unavailable sentinel and is never an evaluated decision.
    """
    if len(dates) < 2 or dates != sorted(set(dates)): raise ValueError('Sorted unique session calendar required')
    records = first_available(records); times = [timestamp(r['available_at']) for r in records]
    result = []
    for i, date in enumerate(dates):
        execution = dates[i + 1] if i + 1 < len(dates) else None
        cutoff = datetime.strptime(execution, '%Y%m%d').replace(hour=8, minute=55, tzinfo=KST) if execution else None
        j = bisect_right(times, cutoff) - 1 if cutoff else -1
        age = (cutoff - times[j]).total_seconds() / 3600 if j >= 0 else None
        valid = j >= 0 and 0 <= age <= 96; record = records[j] if valid else None
        result.append(dict(date=date, execution_date=execution, signal_cutoff=cutoff.isoformat() if cutoff else None,
                           available=valid, report_available_at=record['available_at'] if valid else None,
                           age_hours=age if valid else None, stock_pct=record['stock_pct'] if valid else None,
                           cash_pct=record.get('cash_pct') if valid else None,
                           report_sha256=record['report_sha256'] if valid else None))
    return result


def freeze():
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    for filename, sha in parent['hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Earlier news dependency changed')
    forecasts = json.loads((PRIOR / 'forecast_hashes.json').read_text())
    if len(forecasts) != 198: raise AssertionError('Incomplete legacy forecast manifest')
    for filename, sha in forecasts.items():
        if hashlib.sha256((MODEL_ROOT / filename).read_bytes()).hexdigest() != sha: raise AssertionError('Legacy forecast changed')
    files = [Path(__file__), PRIOR / 'protocol.json', PRIOR / 'joint_bootstrap.csv', PRIOR / 'forecast_hashes.json',
             PRIOR / 'online_choices.json', MACRO_ROOT / 'records.json', MACRO_ROOT / 'daily_signals.json',
             READINESS / 'readiness.json', READINESS / 'cutoff_comparison.json']
    files += sorted((PRIOR / 'portfolios').glob('*.json')) + sorted((GROSS_ROOT / 'portfolios').glob('*.json'))
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['news_weights', 'macro_archive', 'macro_overlay', 'weighted_replay', 'stock_limits', 'week_boundaries',
               'action_amendment', 'planned_audit', 'walkforward_eval', 'walkforward', 'study', 'features', 'replay', 'data']]
    spec = dict(parent=parent, start='20260519', end='20260923', execution_sessions=88, selectors=SELECTORS, cases=CASES,
                comparisons=COMPARISONS, portfolios='36 new morning accounts;12 unchanged close-stock accounts must reproduce all fields.',
                availability='Exact same first-completed report pool,<=execution08:55 KST and age<=96h. Next-session calendar, including weekends/closures, comes from frozen panel. No raw reports, APIs, live database, or future text reinterpretation.',
                policy='Identical original causal score selectors and5/10%base stock targets times recommendation/0.8; top12,R5,NAV5bp band,max3/10fills,sector20%,gross80%,smallcap30%,dated stock limits and corrected completed-week assessment. Mixed news/index/disclosure/prior-allocation signal.',
                controls='Same dates and initially cash accounts. Compare against old fixed60/fixed80, price volatility, old close-based macro policies and own prior20-session morning mean. Smooth excludes current report.',
                bootstrap=dict(blocks=[5, 10], draws=4000, seed=81, alpha=.025),
                family='Retain75 previous shorter-period news comparisons and add3selectors x4cases x6comparisons=72, total147. Separate from full-period heatmap family.',
                gate='Each incremental comparison must pass both adjustedp<0.025 and simultaneous_lower95>0. Report portfolio return/turnover separately. No heatmap promotion or independent-confirmation claim from this overlay experiment.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    # JSON serialises tuple pairs as arrays; store one canonical representation.
    spec = json.loads(json.dumps(spec))
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Morning news protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:03d}_' + p.name))
    return spec


def run():
    spec = freeze(); parent = spec['parent']; action = parent['action']; start, end = spec['start'], spec['end']
    p, ix, ci, di = context(SOURCE); p, release = amend_actions(p, ix, action['actions'])
    dates = np.asarray(ix['dates']); used = np.flatnonzero((dates >= start) & (dates <= end)); signals = used - 1
    expected_dates = dates[used].tolist(); assert len(expected_dates) == 88
    close, _ = schedules_from_records(p, ix['dates'], json.loads((MACRO_ROOT / 'daily_signals.json').read_text()))
    records = morning_signals(json.loads((MACRO_ROOT / 'records.json').read_text()), ix['dates'])
    morning, available = schedules_from_records(p, ix['dates'], records)
    ready = json.loads((READINESS / 'cutoff_comparison.json').read_text())
    for item, d in zip(ready, used):
        r = records[d - 1]
        assert r['date'] == item['signal_date'] and r['execution_date'] == item['execution_date'] == dates[d]
        assert r['report_sha256'] == item['morning_report_sha256'] and r['signal_cutoff'] == item['morning_cutoff']
        assert morning['macro_stock'][d-1] == item['morning_stock_fraction']
        if r['available']: assert timestamp(r['report_available_at']) <= timestamp(r['signal_cutoff'])
    assert len(ready) == 88
    atomic_json(DEST / 'morning_signals.json', records)
    schedules = {'morning_stock': morning['macro_stock'], 'morning_price_min': morning['price_macro_min'],
                 'morning_smooth': morning['macro_smooth']}
    table = pd.DataFrame(dict(signal_date=dates[signals], execution_date=dates[used], available=available[signals]))
    for name, values in schedules.items(): table[name] = values[signals] / .8
    table.to_csv(DEST / 'target_factors.csv', index=False)
    scores, choices, _ = load_scores(MODEL_ROOT, p, ix, ci, di, action['prior']['model_protocol'])
    if choices != json.loads((PRIOR / 'online_choices.json').read_text()): raise AssertionError('Legacy selectors changed')
    atomic_json(DEST / 'online_choices.json', choices)
    p['sector_cap'] = np.minimum(p['sector_cap'], .20); caps = historical_stock_caps(ix['codes'], ix['dates'])
    cache = {}; values = {}

    def account(root, key):
        if (str(root), key) not in cache:
            cache[str(root), key] = json.loads((root / 'portfolios' / (key + '.json')).read_text())
        return cache[str(root), key]

    def returns(root, key):
        if (str(root), key) not in values:
            result = account(root, key)
            if [d['date'] for d in result['daily']] != expected_dates: raise AssertionError('Morning return dates differ')
            nav = np.array([r['nav'] for r in result['daily']]); values[str(root), key] = nav / np.r_[1e9, nav[:-1]] - 1
        return values[str(root), key]

    rows = []; audits = {}; regressions = []; folder = DEST / 'portfolios'; folder.mkdir(exist_ok=True)
    for selector in SELECTORS:
        for case in CASES:
            prefix = selector + '__' + case['id'] + '__'
            kw = dict(top_n=12, weight=case['weight'], rebalance=5, rebalance_band=.0005,
                      max_orders=case['max_orders'], return_trades=True, planning_price='open',
                      action_release_dates=release, stock_cap_schedule=caps)
            control = replay(p, ix, scores[selector], start, end, weight_schedule=case['weight'] * (close['macro_stock'] / .8), **kw)
            control['metrics'].update(assess_weeks(control))
            if control != account(PRIOR, prefix + 'macro_stock'): raise AssertionError('Close-based reference changed')
            regressions.append(dict(id=prefix + 'macro_stock', all_fields_equal=True))
            for arm, schedule in schedules.items():
                key = prefix + arm; path = folder / (key + '.json')
                if path.exists(): raise RuntimeError('Do not overwrite morning account')
                result = replay(p, ix, scores[selector], start, end, weight_schedule=case['weight'] * (schedule / .8), **kw)
                result['metrics'].update(assess_weeks(result)); atomic_json(path, result)
                audits[key] = checked(p, ix, result, release, case['max_orders'])
                quarters = calendar_blocks(result['daily'])
                rows.append(dict(id=key, selector=selector, case=case['id'], arm=arm, **result['metrics'],
                                 **quarters, positive_blocks=sum(v > 0 for v in quarters.values())))
            print(json.dumps(dict(selector=selector, case=case['id'], portfolios=len(rows))), flush=True)
    assert len(rows) == 36 and len(regressions) == 12
    pd.DataFrame(rows).to_csv(DEST / 'portfolio_summary.csv', index=False)
    atomic_json(DEST / 'independent_audit.json', audits); atomic_json(DEST / 'close_regression.json', regressions)
    previous = pd.read_csv(PRIOR / 'joint_bootstrap.csv'); family = []; differences = []; error = 0.
    columns = ['origin', 'selector', 'case', 'treatment', 'control']
    for row in previous[previous.block == 5].to_dict('records'):
        h = {k: row[k] for k in columns}
        if h['origin'] == 'prior':
            root = GROSS_ROOT; tk = h['selector'] + '__' + h['treatment'] + '__' + h['case']; ck = h['selector'] + '__' + h['control'] + '__' + h['case']
        elif h['origin'] == 'new':
            root = PRIOR; prefix = h['selector'] + '__' + h['case'] + '__'; tk, ck = prefix + h['treatment'], prefix + h['control']
        else: raise AssertionError('Unknown retained macro origin')
        diff = returns(root, tk) - returns(root, ck); error = max(error, abs(float(diff.mean()) - row['mean']))
        family.append(h); differences.append(diff)
    assert len(family) == 75 and error < 1e-14
    atomic_json(DEST / 'prior_family_reconstruction.json', dict(hypotheses=75, maximum_mean_error=error))
    paired = []
    for selector in SELECTORS:
        for case in CASES:
            prefix = selector + '__' + case['id'] + '__'
            for treatment, control in COMPARISONS:
                root = DEST if control.startswith('morning_') else PRIOR; tk, ck = prefix + treatment, prefix + control
                h = dict(origin='morning', selector=selector, case=case['id'], treatment=treatment, control=control)
                family.append(h); differences.append(returns(DEST, tk) - returns(root, ck))
                a, b = account(DEST, tk), account(root, ck)
                paired.append(dict(h, treatment_id=tk, control_id=ck, trades_changed=a['trades'] != b['trades'],
                                   return_difference=a['metrics']['return'] - b['metrics']['return'],
                                   gross_difference=a['metrics']['mean_gross'] - b['metrics']['mean_gross'],
                                   treatment_turnover_fail=a['metrics']['four_week_turnover_stop']))
    assert len(family) == 147 and len({tuple(h[k] for k in columns) for h in family}) == 147
    pd.DataFrame(paired).to_csv(DEST / 'paired_effects.csv', index=False); statistics = []
    for block in [5, 10]:
        boot = family_bootstrap(np.column_stack(differences), block=block, draws=4000, seed=81)
        for i, h in enumerate(family):
            statistics.append(dict(h, block=block, **{k: float(boot[k][i]) for k in
                              ['mean', 'standard_error', 'adjusted_p', 'marginal_p', 'simultaneous_lower95']}))
    stats = pd.DataFrame(statistics); stats.to_csv(DEST / 'joint_bootstrap.csv', index=False); passed = []
    for h in family:
        inference = [r for r in statistics if all(r[k] == v for k, v in h.items())]
        assert len(inference) == 2
        if all(r['adjusted_p'] < .025 and r['simultaneous_lower95'] > 0 for r in inference): passed.append(h)
    result = dict(status='development_only', portfolios=36, all_fields_close_regressions=12, execution_sessions=88,
                  joint_incremental_news_hypotheses=147, passed_incremental_effects=passed,
                  new_pairs_with_changed_trades=sum(r['trades_changed'] for r in paired),
                  minimum_adjusted_p=min(r['adjusted_p'] for r in statistics),
                  heatmap_promotion_assessed=False, independent_confirmation=False)
    atomic_json(DEST / 'evaluation_summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

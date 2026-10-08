"""Descriptive signal and fixed-fill cost attribution; never a tradable backtest.

Top-minus-bottom is a cross-sectional label statistic, not a short account.
No threshold candidate or significance claim may be generated from this output.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    out = np.empty(len(values), float)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        out[order[first:last]] = (first + last - 1) / 2
        first = last
    return out


def cross_section(score, outcome, eligible):
    valid = eligible & np.isfinite(score) & np.isfinite(outcome)
    ids = np.flatnonzero(valid)
    if len(ids) < 20:
        return None
    # Stable code order resolves equal scores without consulting outcomes.
    order = ids[np.argsort(score[ids], kind='stable')]
    n = max(1, len(order) // 10)
    x, y = ranks(score[ids]), ranks(outcome[ids])
    ic = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.
    market = float(outcome[ids].mean())
    top, bottom = float(outcome[order[-n:]].mean()), float(outcome[order[:n]].mean())
    return dict(n=len(ids), top_n=n, rank_ic=ic, universe_return=market,
                top_return=top, bottom_return=bottom, top_excess=top-market,
                bottom_excess=bottom-market, top_minus_bottom=top-bottom)


def forward_outcomes(panel, day, horizon):
    """Next execution-window price to the same window h sessions later.

    Actions in the interval are excluded from this diagnostic so neither locked
    entitlements nor inferred action cash flows become freely tradable labels.
    This future-availability screen is explicitly descriptive, not a buy filter.
    """
    first, last = day + 1, day + 1 + horizon
    price = panel['exec_price']
    if last >= price.shape[1]:
        return None
    good = (np.isfinite(price[:, first]) & (price[:, first] > 0)
            & np.isfinite(price[:, last]) & (price[:, last] > 0)
            & (panel['exec_count'][:, first] >= 25)
            & (panel['exec_count'][:, last] >= 25)
            & np.all(np.abs(panel['split'][:, first:last+1]-1) <= .002, axis=1))
    result = np.full(price.shape[0], np.nan)
    result[good] = price[good, last] / price[good, first] - 1
    return result


def nav_stats(daily, nav):
    nav = np.asarray(nav, float)
    ret = nav / np.r_[1e9, nav[:-1]] - 1
    def sharpe(a):
        return float(np.mean(a) / np.std(a, ddof=1) * np.sqrt(252)) if len(a)>1 and np.std(a, ddof=1)>0 else 0.
    monthly = {m: sharpe(ret[[d['date'][:6] == m for d in daily]])
               for m in sorted({d['date'][:6] for d in daily})}
    return dict(pooled_sharpe=sharpe(ret), minimum_fold_sharpe=min(monthly.values()),
                monthly_sharpe=monthly, total_return=float(nav[-1]/1e9-1))


def fixed_fill_costs(result, panel, index):
    """Add paid fees and modeled impact back as idle cash, with fills fixed."""
    codes = {c:i for i,c in enumerate(index['codes'])}
    dates = {d:i for i,d in enumerate(index['dates'])}
    costs = {d['date']:0. for d in result['daily']}
    fee_total = slip_total = 0.
    for t in result['trades']:
        mid = float(panel['exec_price'][codes[t['code']], dates[t['date']]])
        slip = t['qty'] * ((t['price']-mid) if t['side']=='buy' else (mid-t['price']))
        assert slip >= -.00001
        costs[t['date']] += t['fee']+slip
        fee_total += t['fee']; slip_total += slip
    assert abs(fee_total-result['metrics']['fees_krw']) < .01
    assert abs(slip_total-result['metrics']['slippage_krw']) < .01
    daily = result['daily']; nav = np.array([d['nav'] for d in daily])
    no_cost = nav + np.cumsum([costs[d['date']] for d in daily])
    return dict(net=nav_stats(daily, nav), fixed_fill_cost_added_back=nav_stats(daily, no_cost),
                fees_krw=fee_total, slippage_krw=slip_total,
                cost_fraction_initial_nav=(fee_total+slip_total)/1e9,
                cost_by_date=costs,
                caveat='Same fills and positions, paid costs returned to idle cash; not a zero-cost reoptimization or admissible candidate.')


def run(market, source, output):
    from quant.timefolio_heatmap_fleet_accounts import load_market
    from quant.timefolio_heatmap_gpu_worker import verify, digest, write
    assert not output.exists()
    verify(market)
    panel, ix, _, _, _ = load_market(market)
    hashes = json.loads((source/'input_hashes.json').read_text())
    for name, sha in hashes.items():
        assert digest(source/name) == sha
    output.mkdir(parents=True)
    records, inputs = [], {}
    cases = json.loads((source/'design_registration.json').read_text())['source_cases']
    outcomes = {(d,h):forward_outcomes(panel,d,h) for d,date in enumerate(ix['dates'])
                if '20260101' <= date <= '20260923' for h in [1,3,5,10]}
    for case in cases:
        for p in sorted((source/'sources'/case/'scores').glob('*.npy')):
            score = np.load(p, allow_pickle=False); inputs[str(p)] = digest(p)
            for (d,h), outcome in outcomes.items():
                if outcome is None: continue
                row = cross_section(score[:,d], outcome, panel['eligible'][:,d])
                if row:
                    records.append(dict(model=p.stem,case=case,
                        objective='untrained' if '_untrained_' in p.stem else 'trained',
                        member=p.stem.rsplit('_',1)[1],date=ix['dates'][d],horizon=h,**row))
    with (output/'signal_daily.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    groups={}
    for row in records:
        for month in ['pooled',row['date'][:6]]:
            groups.setdefault((row['model'],row['horizon'],month),[]).append(row)
    summary=[]
    fields=['rank_ic','universe_return','top_return','bottom_return','top_excess','bottom_excess','top_minus_bottom']
    for (model,h,month),rows in sorted(groups.items()):
        summary.append(dict(model=model,horizon=h,month=month,days=len(rows),
            **{key:float(np.mean([r[key] for r in rows])) for key in fields},
            positive_top_excess_days=sum(r['top_excess']>0 for r in rows),
            positive_ic_days=sum(r['rank_ic']>0 for r in rows)))
    write(output/'signal_summary.json',summary)
    accounts=[]
    for case in cases:
        for p in sorted((source/'sources'/case/'portfolios').glob('*.json')):
            result=json.loads(p.read_text()); attribution=fixed_fill_costs(result,panel,ix)
            accounts.append(dict(id=p.stem,case=case,source_sha256=digest(p),
                **attribution,metrics=result['metrics']))
    assert len(accounts)==1152
    write(output/'fixed_fill_attribution.json',accounts)
    write(output/'complete.json',dict(at=time.time(),accounts=len(accounts),scores=len(inputs),
        signal_rows=len(records),source_scores=inputs,source_manifest_sha256=digest(source/'input_hashes.json'),
        code_sha256=digest(__file__),descriptive_only=True,short_accounts_created=0,
        new_strategy_accounts=0,new_hypotheses=0,reserved_outcomes_read=False,
        selection_bias='Previously selected nine source models; overlapping horizons and reused development outcomes; no significance claim.',
        label_missingness='Future execution availability/action screen; label coverage reported; not used for strategy admission.',
        hashes={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    print(json.dumps(dict(accounts=len(accounts),signal_rows=len(records),output=str(output))),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ['market','source','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.market,a.source,a.output)

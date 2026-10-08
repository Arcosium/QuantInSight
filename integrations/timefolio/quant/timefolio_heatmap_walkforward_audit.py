"""Independent fill reconstruction for the monthly research batch."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from quant.timefolio_heatmap_audit import audit_fills
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward_eval import DEST, SOURCE, market_regimes


def additional_checks(p, ix, result, schedule, *, participation=.05, slip=.0005, max_orders=20):
    codes = {code:i for i,code in enumerate(ix['codes'])}; dates = {date:i for i,date in enumerate(ix['dates'])}
    grouped = {}; errors = []; qty = np.zeros(len(codes)); cash = 1e9; marks = np.zeros(len(codes))
    for t in result['trades']: grouped.setdefault(t['date'],[]).append(t)
    max_count = 0; sector_exposure = {int(s):[] for s in np.unique(p['sector'])}
    for row in result['daily']:
        d = dates[row['date']]; ratio = p['split'][:,d]; new = qty*ratio
        cash += np.sum(np.where(qty > 0,(new-np.floor(new))*np.nan_to_num(p['close'][:,d]),0))
        qty = np.floor(new+1e-7); marks = np.divide(marks,ratio,out=marks.copy(),where=ratio>0)
        price = p['exec_price'][:,d].astype(float); available = np.isfinite(price)&(price>0)&(p['exec_count'][:,d]>=25)
        marks[available] = price[available]
        trades = grouped.get(row['date'],[]); max_count=max(max_count,len(trades))
        if len(trades)>max_orders: errors.append({'date':row['date'],'kind':'daily_order_budget'})
        volume = Counter()
        for t in trades:
            i=codes[t['code']]; buy=t['side']=='buy'; q=t['qty']; volume[i]+=q
            if t['signal_date'] != ix['dates'][d-1] or not (q>0 and int(q)==q):
                errors.append({'date':row['date'],'kind':'signal_or_quantity'})
            expected=price[i]*(1+slip if buy else 1-slip)
            if not np.isclose(t['price'],expected,rtol=1e-10,atol=1e-8): errors.append({'date':row['date'],'kind':'fill_price'})
            fee=q*t['price']*(.001 if buy else .003)
            if not np.isclose(t['fee'],fee,rtol=1e-10,atol=1e-6): errors.append({'date':row['date'],'kind':'fee'})
            qty[i]+=q if buy else -q; cash+=(-q*t['price'] if buy else q*t['price'])-t['fee']
            if cash<-.01 or np.any(qty<0): errors.append({'date':row['date'],'kind':'cash_or_short'})
            value=qty@marks; nav=cash+value
            ceiling=.8 if schedule is None else schedule[d-1]
            if buy and value>ceiling*nav+1: errors.append({'date':row['date'],'kind':'dynamic_gross'})
        for i,q in volume.items():
            if q>np.floor(p['exec_volume'][i,d]*participation): errors.append({'date':row['date'],'kind':'participation'})
        close=p['close'][:,d]; good=np.isfinite(close)&(close>0); marks[good]=close[good]
        values=qty*marks; nav=cash+values.sum()
        for sec in sector_exposure: sector_exposure[sec].append(float(values[p['sector']==sec].sum()/nav))
    return {'additional_errors':errors,'observed_max_daily_orders':max_count,
            'mean_account_sector_weights':{str(s):float(np.mean(v)) for s,v in sector_exposure.items()}}


def audit(dest=DEST,source=SOURCE):
    dest,source=Path(dest),Path(source);p,ix,_,_=context(source);schedules,_=market_regimes(p)
    spec=json.loads((dest/'protocol.json').read_text());policies={x['id']:x for x in spec['policies']};results={}
    for path in sorted((dest/'portfolios').glob('*.json')):
        name,policy_name=path.stem.split('__');policy=policies[policy_name]
        panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],policy['sector_cap'])
        result=json.loads(path.read_text())
        check=audit_fills(panel,ix,result)
        check.update(additional_checks(panel,ix,result,schedules[policy['regime']]))
        results[path.stem]=check
    summary={'portfolios':len(results),'buy_checks':sum(r['buy_checks'] for r in results.values()),
             'post_buy_violations':sum(len(r['post_buy_limit_violations']) for r in results.values()),
             'additional_errors':sum(len(r['additional_errors']) for r in results.values()),
             'max_nav_error_krw':max((r['maximum_nav_reconstruction_error_krw'] for r in results.values()),default=0)}
    atomic_json(dest/'independent_audit.json',{'summary':summary,'portfolios':results})
    print(json.dumps(summary),flush=True)
    if summary['post_buy_violations'] or summary['additional_errors'] or summary['max_nav_error_krw']>.01:
        raise AssertionError('Independent audit failed; inspect before reporting')
    return summary


if __name__=='__main__':audit()

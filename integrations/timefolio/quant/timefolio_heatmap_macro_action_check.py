"""Mandatory action-ledger correction of the unchanged 27 macro comparisons."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import DEST as ACTION_ROOT,amend_actions,release_audit
from quant.timefolio_heatmap_macro_amendment import DEST as PRIOR,COMPARISONS
from quant.timefolio_heatmap_macro_overlay import MACRO_ROOT,schedules_from_records
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores,family_bootstrap
from quant.timefolio_heatmap_replay import INITIAL_NAV

DEST=SOURCE.with_name('20260928_macro_overlay_v3')


def run():
    prior=json.loads((PRIOR/'protocol.json').read_text());action=json.loads((ACTION_ROOT/'protocol.json').read_text())
    spec={'parent':prior,'action_correction':action,'selection':'same selectors, schedules, budgets, dates and joint 27-hypothesis family; only mandatory ledger corrections',
          'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'forecast_hashes':json.loads((ACTION_ROOT/'forecast_hashes.json').read_text())}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Macro action check changed')
    if not path.exists():atomic_json(path,spec);shutil.copy2(Path(__file__),DEST/Path(__file__).name)
    p,ix,ci,di=context(SOURCE);p,release=amend_actions(p,ix,action['actions']);p['sector_cap']=np.minimum(p['sector_cap'],.15)
    schedules,available=schedules_from_records(p,ix['dates'],json.loads((MACRO_ROOT/'daily_signals.json').read_text()))
    scores,_,_=load_scores(MODEL_ROOT,p,ix,ci,di,action['prior']['model_protocol'])
    output=DEST/'portfolios';output.mkdir(exist_ok=True);rows=[];returns={};audits={}
    for budget in prior['order_budgets']:
        for selector in prior['selectors']:
            for arm,schedule in schedules.items():
                key=f'{selector}__{arm}__orders{budget}';path=output/(key+'.json')
                if path.exists():r=json.loads(path.read_text())
                else:
                    r=replay(p,ix,scores[selector],prior['start'],prior['end'],top_n=24,weight=.04,
                             gross_schedule=schedule,max_orders=budget,return_trades=True,
                             planning_price='open',action_release_dates=release);atomic_json(path,r)
                check=audit_fills(p,ix,r);check.update(additional_checks(p,ix,r,schedule,max_orders=budget))
                check['announced_action_errors']=release_audit(p,ix,r,release)
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['maximum_nav_reconstruction_error_krw']>.01):raise AssertionError('Corrected macro audit failed')
                audits[key]=check;nav=np.array([x['nav'] for x in r['daily']]);returns[key]=nav/np.r_[INITIAL_NAV,nav[:-1]]-1
                rows.append(dict(id=key,selector=selector,arm=arm,order_budget=budget,**r['metrics']))
    statistics=[];hypotheses=[];differences=[]
    for budget in prior['order_budgets']:
        for selector in prior['selectors']:
            for treatment,control in COMPARISONS:
                hypotheses.append(dict(selector=selector,order_budget=budget,treatment=treatment,control=control))
                differences.append(returns[f'{selector}__{treatment}__orders{budget}']-returns[f'{selector}__{control}__orders{budget}'])
    for block in [5,10]:
        result=family_bootstrap(np.column_stack(differences),block=block,draws=4000,seed=81)
        for i,h in enumerate(hypotheses):statistics.append(dict(h,block=block,**{k:float(result[k][i]) for k in
                                                          ['mean','standard_error','adjusted_p','simultaneous_lower95']}))
    pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
    pd.DataFrame(statistics).to_csv(DEST/'bootstrap.csv',index=False);atomic_json(DEST/'independent_audit.json',audits)
    dates=np.asarray(ix['dates']);used=(dates>=prior['start'])&(dates<=prior['end']);signals=np.flatnonzero(used)-1
    binding=int(np.sum(schedules['price_macro_min'][signals]<schedules['price_vol'][signals]-1e-12))
    summary={'status':'development_only','portfolios':len(rows),'hypotheses':len(hypotheses),
             'covered_signal_sessions':int(available[signals].sum()),'macro_reduced_price_vol_target_sessions':binding,
             'minimum_adjusted_p':min(r['adjusted_p'] for r in statistics),
             'passed_hypotheses':[h for h in hypotheses if all(r['adjusted_p']<.05 and r['simultaneous_lower95']>0 for r in statistics if all(r[k]==v for k,v in h.items()))]}
    atomic_json(DEST/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':run()

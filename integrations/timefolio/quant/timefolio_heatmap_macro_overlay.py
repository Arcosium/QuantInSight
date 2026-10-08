"""Predeclared exploratory macro overlay on causal monthly model selectors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_replay import replay, INITIAL_NAV
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, market_regimes, family_bootstrap
from quant.timefolio_heatmap_audit import audit_fills
from quant.timefolio_heatmap_walkforward_audit import additional_checks
from quant.timefolio_heatmap_macro_archive import timestamp

MACRO_ROOT=SOURCE.with_name('20260928_macro_archive_v2')
DEST=SOURCE.with_name('20260928_macro_overlay_v1')
SELECTORS=['online_neural','online_neural3','online_nonimage']


def schedules_from_records(p,dates,records):
    if [r['date'] for r in records] != dates:raise ValueError('Macro and price date axes differ')
    macro=np.full(len(dates),.6);available=np.zeros(len(dates),bool)
    for i,row in enumerate(records):
        if not row['available']:continue
        if timestamp(row['report_available_at']) > timestamp(row['signal_cutoff']):raise ValueError('Future macro report')
        if row['age_hours'] < 0 or row['age_hours'] > 96:raise ValueError('Stale macro report')
        if row['stock_pct'] is None or not 0<=row['stock_pct']<=1:raise ValueError('Missing stock recommendation')
        macro[i]=np.clip(row['stock_pct'],.3,.8);available[i]=True
    regimes,_=market_regimes(p)
    smooth=pd.Series(macro).shift(1).rolling(20,min_periods=1).mean().fillna(.6).to_numpy()
    return {'fixed60':np.full(len(dates),.6),'price_vol':regimes['volatility'],
            'macro_stock':macro,'price_macro_min':np.minimum(regimes['volatility'],macro),
            'macro_smooth':smooth},available


def freeze():
    DEST.mkdir(exist_ok=True)
    paths=[Path(__file__),MACRO_ROOT/'daily_signals.json',MACRO_ROOT/'readiness.json',MODEL_ROOT/'protocol.json',
           Path(__file__).with_name('timefolio_heatmap_walkforward_eval.py'),Path(__file__).with_name('timefolio_heatmap_replay.py')]
    spec={'version':'macro-overlay-v1','status':'development_only; history reused',
          'start':'20260519','end':'20260923','selectors':SELECTORS,
          'arms':['fixed60','price_vol','macro_stock','price_macro_min','macro_smooth'],
          'macro':'first completed KR report; close 15:30 KST; 96-hour maximum age; stock recommendation clipped to [0.3,0.8]; unavailable = 0.6',
          'smooth':'mean of previous 20 signal-day macro exposures, excludes current day, initial 0.6',
          'comparisons':[['price_macro_min','price_vol'],['macro_stock','fixed60'],['macro_stock','macro_smooth']],
          'statistics':'single family of nine paired daily excess-return hypotheses; one-sided circular-block max-t, blocks5/10, 4000 draws, seed81',
          'sector_cap':.15,'stock_weight':.04,'top_n':24,'max_daily_orders':20,'gross_ceiling':.8,
          'retraining':'none; same frozen monthly causal selectors for every arm',
          'limitations':'mixed news/index/disclosure/prior-allocation signal; not pure news effect; prompt/model changes; several account streams; short reused history',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Macro protocol changed')
    if not path.exists():atomic_json(path,spec)
    return spec


def run():
    spec=freeze();p,ix,ci,di=context(SOURCE)
    records=json.loads((MACRO_ROOT/'daily_signals.json').read_text())
    schedules,available=schedules_from_records(p,ix['dates'],records)
    model_spec=json.loads((MODEL_ROOT/'protocol.json').read_text())
    scores,choices,_=load_scores(MODEL_ROOT,p,ix,ci,di,model_spec)
    panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],spec['sector_cap'])
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True);rows=[];returns={};audits={}
    for selector in SELECTORS:
        for arm,schedule in schedules.items():
            key=selector+'__'+arm;path=folder/(key+'.json')
            if path.exists():result=json.loads(path.read_text())
            else:
                result=replay(panel,ix,scores[selector],spec['start'],spec['end'],top_n=24,weight=.04,
                              gross_schedule=schedule,max_orders=20,return_trades=True)
                atomic_json(path,result)
            checks=audit_fills(panel,ix,result);checks.update(additional_checks(panel,ix,result,schedule))
            audits[key]=checks
            if checks['post_buy_limit_violations'] or checks['additional_errors'] or checks['maximum_nav_reconstruction_error_krw']>.01:
                raise AssertionError('Macro overlay audit failed')
            nav=np.array([r['nav'] for r in result['daily']]);returns[key]=nav/np.r_[INITIAL_NAV,nav[:-1]]-1
            rows.append({'id':key,'selector':selector,'arm':arm,**result['metrics']})
        print(json.dumps({'macro_selector_complete':selector}),flush=True)
    hypotheses=[];differences=[]
    for selector in SELECTORS:
        for treatment,control in spec['comparisons']:
            hypotheses.append({'selector':selector,'treatment':treatment,'control':control})
            differences.append(returns[selector+'__'+treatment]-returns[selector+'__'+control])
    statistics=[]
    for block in [5,10]:
        result=family_bootstrap(np.column_stack(differences),block=block,draws=4000,seed=81)
        for i,hypothesis in enumerate(hypotheses):
            statistics.append(dict(hypothesis,block=block,**{key:float(result[key][i]) for key in
                              ['mean','standard_error','adjusted_p','simultaneous_lower95']}))
    pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
    pd.DataFrame(statistics).to_csv(DEST/'bootstrap.csv',index=False)
    atomic_json(DEST/'independent_audit.json',audits)
    atomic_json(DEST/'summary.json',{'status':'development_only','portfolios':len(rows),'hypotheses':len(hypotheses),
                                   'covered_signal_sessions':int(available.sum()),'statistics':statistics})
    print(json.dumps({'macro_overlay_complete':True,'portfolios':len(rows)}),flush=True)


if __name__=='__main__':run()

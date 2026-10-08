"""Repeat fixed execution probes with causal order plans and a verified action."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE,context,monthly_folds
from quant.timefolio_heatmap_capacity_probe import MODELS,CASES
from quant.timefolio_heatmap_study import score_matrix
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks

DEST=SOURCE.with_name('20260928_planning_probe_v1')
ACTION={'code':'068270','ex_date':'20260604','ratio':1.05,'listing_date':'20260630',
        'announcement':'20260521','new_shares':10920342,
        'shares_20260629':221633364,'shares_20260630':232553706,
        'sources':['https://www.celltrion.com/ko-kr/investment/notice/announce/4775',
                   'https://www.celltrion.com/ko-kr/company/notice/4784'],
        'confirmation':'Historical KRX listed shares increase by exactly 10,920,342 on June 30; source parquet retains integer precision.'}


def announced_action_check(panel,ix,result):
    i=ix['codes'].index(ACTION['code']);by_date={};qty=0;locked=0;errors=[]
    for trade in result['trades']:
        if trade['code']==ACTION['code']:by_date.setdefault(trade['date'],[]).append(trade)
    index={d:i for i,d in enumerate(ix['dates'])}
    for row in result['daily']:
        d=index[row['date']];old=qty;qty=int(np.floor(qty*panel['split'][i,d]+1e-7))
        if row['date']==ACTION['ex_date']:locked+=qty-old
        if row['date']>=ACTION['listing_date']:locked=0
        for trade in by_date.get(row['date'],[]):
            if trade['side']=='buy':qty+=trade['qty']
            else:
                if trade['qty']>qty-locked:errors.append({'date':row['date'],'kind':'announced_bonus_sold_before_listing'})
                qty-=trade['qty']
    return errors


def run():
    DEST.mkdir(exist_ok=True)
    files=[Path(__file__),Path(__file__).with_name('timefolio_heatmap_planned_replay.py'),Path(__file__).with_name('timefolio_heatmap_planned_audit.py')]
    spec={'models':MODELS,'cases':CASES,'policy':'sector15 stock4% top24 gross80% rebalance5 Jan-Sep2026',
          'clocks':['open','previous_close'],'previous_close_cases':['base','orders3'],
          'uncorrected_action_control':'open clock, base/orders3; original inferred ratio and release heuristic',
          'fractional_entitlements':'omitted conservatively from cash and NAV; not credited before actual settlement',
          'action':ACTION,'interpretation':'methodology sensitivity on already observed development data; no candidate chosen by this probe',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Planning specification changed')
    if not path.exists():atomic_json(path,spec)
    p,ix,ci,di=context(SOURCE);panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],.15)
    corrected=dict(panel);corrected['split']=p['split'].astype(float).copy()
    corrected['split'][ix['codes'].index(ACTION['code']),ix['dates'].index(ACTION['ex_date'])]=ACTION['ratio']
    release={(ACTION['code'],ACTION['ex_date']):ACTION['listing_date']}
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True);rows=[];audits={}
    for name in MODELS:
        if name=='lowvol':scores=-p['vol20']
        else:
            values=np.full(len(ci),np.nan,np.float32)
            for fold in monthly_folds(ix['dates']):
                a=np.load(MODEL_ROOT/'models'/name/(fold['month']+'.pred.npy'));mask=np.isfinite(a);values[mask]=a[mask]
            scores=score_matrix(values,ci,di,p['close'].shape)
        for clock in ['open','previous_close']:
            for corrected_action in [True,False]:
                if not corrected_action and clock!='open':continue
                actual=corrected if corrected_action else panel
                for case in CASES:
                    if (clock=='previous_close' or not corrected_action) and case['id'] not in ['base','orders3']:continue
                    key=f"{name}__{clock}__{'verified_action' if corrected_action else 'inferred_action'}__{case['id']}"
                    path=folder/(key+'.json');kwargs={k:v for k,v in case.items() if k!='id'}
                    if path.exists():result=json.loads(path.read_text())
                    else:
                        result=replay(actual,ix,scores,'20260101','20260923',weight=.04,top_n=24,return_trades=True,
                                      planning_price=clock,action_release_dates=release if corrected_action else None,**kwargs)
                        atomic_json(path,result)
                    check=audit_fills(actual,ix,result);check.update(additional_checks(actual,ix,result,None,**kwargs))
                    check['announced_action_errors']=announced_action_check(actual,ix,result) if corrected_action else []
                    audits[key]=check
                    if check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors'] or check['maximum_nav_reconstruction_error_krw']>.01:
                        raise AssertionError('Corrected execution audit failed')
                    rows.append(dict(model=name,clock=clock,verified_action=corrected_action,scenario=case['id'],**result['metrics']))
                    print(json.dumps({'model':name,'clock':clock,'verified_action':corrected_action,'scenario':case['id'],
                                      'return':result['metrics']['return'],'mdd':result['metrics']['mdd'],
                                      'gross':result['metrics']['mean_gross'],'turnover_failed':result['metrics']['four_week_turnover_stop']}),flush=True)
    pd.DataFrame(rows).to_csv(DEST/'summary.csv',index=False);atomic_json(DEST/'independent_audit.json',audits)


if __name__=='__main__':run()

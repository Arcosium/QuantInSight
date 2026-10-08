"""Execution stress for fixed reference models, without choosing a new strategy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE,context,monthly_folds
from quant.timefolio_heatmap_study import score_matrix
from quant.timefolio_heatmap_replay import replay
from quant.timefolio_heatmap_audit import audit_fills
from quant.timefolio_heatmap_walkforward_audit import additional_checks

DEST=SOURCE.with_name('20260928_capacity_probe_v1')
MODELS=['monthly_binary','cnn_sector_h3','cnn_sector_h5','lowvol']
CASES=[dict(id='base',max_orders=20,participation=.05,slip=.0005),
       dict(id='orders10',max_orders=10,participation=.05,slip=.0005),
       dict(id='orders3',max_orders=3,participation=.05,slip=.0005),
       dict(id='participation1',max_orders=20,participation=.01,slip=.0005),
       dict(id='slippage25bp',max_orders=20,participation=.05,slip=.0025),
       dict(id='combined',max_orders=3,participation=.01,slip=.0025)]


def run():
    DEST.mkdir(exist_ok=True)
    spec={'models':MODELS,'cases':CASES,'policy':'sector15; stock4%; top24; gross80%; rebalance5; Jan-Sep 2026',
          'purpose':'fixed reference-model execution sensitivity; not strategy selection or statistical alpha proof',
          'three_order_interpretation':'30-minute window, hypothetical 10-minute cooldown after each fill; only an order-count approximation, not reconstructed depth, queue or exact execution prices',
          'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Capacity protocol changed')
    if not path.exists():atomic_json(path,spec)
    p,ix,ci,di=context(SOURCE);panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],.15)
    summary=[];audits={};folder=DEST/'portfolios';folder.mkdir(exist_ok=True)
    for name in MODELS:
        if name=='lowvol':scores=-p['vol20']
        else:
            values=np.full(len(ci),np.nan,np.float32)
            for fold in monthly_folds(ix['dates']):
                a=np.load(MODEL_ROOT/'models'/name/(fold['month']+'.pred.npy'));mask=np.isfinite(a);values[mask]=a[mask]
            scores=score_matrix(values,ci,di,p['close'].shape)
        for case in CASES:
            key=name+'__'+case['id'];path=folder/(key+'.json')
            kwargs={k:v for k,v in case.items() if k!='id'}
            if path.exists():result=json.loads(path.read_text())
            else:
                result=replay(panel,ix,scores,'20260101','20260923',weight=.04,top_n=24,return_trades=True,**kwargs)
                atomic_json(path,result)
            a=audit_fills(panel,ix,result);a.update(additional_checks(panel,ix,result,None,**kwargs));audits[key]=a
            if a['post_buy_limit_violations'] or a['additional_errors'] or a['maximum_nav_reconstruction_error_krw']>.01:
                raise AssertionError('Execution audit failed')
            summary.append(dict(model=name,scenario=case['id'],**result['metrics']))
            print(json.dumps({'model':name,'scenario':case['id'],'return':result['metrics']['return'],
                              'mdd':result['metrics']['mdd'],'gross':result['metrics']['mean_gross'],
                              'turnover_failed':result['metrics']['four_week_turnover_stop']}),flush=True)
    pd.DataFrame(summary).to_csv(DEST/'summary.csv',index=False)
    atomic_json(DEST/'independent_audit.json',audits)


if __name__=='__main__':run()

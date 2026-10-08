"""Macro overlay repeated with the amended causal order planner, three budgets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_execution_amendment import corrected_panel, BUDGETS
from quant.timefolio_heatmap_macro_overlay import MACRO_ROOT, SELECTORS, schedules_from_records
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_planning_probe import announced_action_check
from quant.timefolio_heatmap_replay import INITIAL_NAV
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, family_bootstrap

DEST = SOURCE.with_name('20260928_macro_overlay_v2')
COMPARISONS = [('price_macro_min', 'price_vol'), ('macro_stock', 'fixed60'), ('macro_stock', 'macro_smooth')]


def freeze():
    files = [Path(__file__), MACRO_ROOT/'daily_signals.json', MODEL_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+name+'.py') for name in
              ['execution_amendment', 'macro_overlay', 'macro_archive', 'planned_replay', 'planned_audit',
               'planning_probe', 'walkforward_eval', 'replay', 'study', 'walkforward', 'features', 'data']]
    spec = {'version': 'macro-overlay-v2', 'start': '20260519', 'end': '20260923',
            'selectors': SELECTORS, 'order_budgets': BUDGETS,
            'arms': ['fixed60', 'price_vol', 'macro_stock', 'price_macro_min', 'macro_smooth'],
            'comparisons': [list(x) for x in COMPARISONS],
            'planning': '09:00 open before execution window; announced Celltrion bonus ratio/listing; no fractional cash',
            'macro': 'same frozen archive and causal schedules as v1; no new training or text interpretation',
            'policy': 'sector15 stock4% top24 gross ceiling80%; 5bp slippage; 5% window participation',
            'family': 'three selectors x three budgets x three paired comparisons = 27; centred circular-block max-t',
            'bootstrap': {'blocks': [5, 10], 'draws': 4000, 'seed': 81},
            'interpretation': 'development-only mixed macro signal, not pure news; prompt/model changes and limited reused observations',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Macro amendment changed')
    if not path.exists():
        atomic_json(path, spec); frozen=DEST/'frozen_source'; frozen.mkdir(exist_ok=True)
        for p in files: shutil.copy2(p, frozen/p.name)
    return spec


def run():
    spec=freeze(); p,ix,ci,di=context(SOURCE); p,release=corrected_panel(p,ix)
    schedules,available=schedules_from_records(p,ix['dates'],json.loads((MACRO_ROOT/'daily_signals.json').read_text()))
    scores,choices,_=load_scores(MODEL_ROOT,p,ix,ci,di,json.loads((MODEL_ROOT/'protocol.json').read_text()))
    atomic_json(DEST/'online_choices.json',choices)
    p['sector_cap']=np.minimum(p['sector_cap'],.15)
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True); rows=[]; returns={}; audits={}
    for budget in BUDGETS:
        for selector in SELECTORS:
            for arm,schedule in schedules.items():
                key=f'{selector}__{arm}__orders{budget}'; path=folder/(key+'.json')
                if path.exists(): result=json.loads(path.read_text())
                else:
                    result=replay(p,ix,scores[selector],spec['start'],spec['end'],top_n=24,weight=.04,
                                  gross_schedule=schedule,max_orders=budget,return_trades=True,
                                  planning_price='open',action_release_dates=release)
                    atomic_json(path,result)
                check=audit_fills(p,ix,result)
                check.update(additional_checks(p,ix,result,schedule,max_orders=budget))
                check['announced_action_errors']=announced_action_check(p,ix,result)
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['maximum_nav_reconstruction_error_krw']>.01):
                    atomic_json(DEST/'failed_audit.json',{'id':key,'audit':check});raise AssertionError('Macro audit failed')
                audits[key]=check
                nav=np.array([r['nav'] for r in result['daily']]);returns[key]=nav/np.r_[INITIAL_NAV,nav[:-1]]-1
                rows.append(dict(id=key,selector=selector,arm=arm,order_budget=budget,**result['metrics']))
            print(json.dumps({'selector':selector,'orders':budget,'portfolios':len(rows)}),flush=True)
    hypotheses=[];differences=[]
    for budget in BUDGETS:
        for selector in SELECTORS:
            for treatment,control in COMPARISONS:
                hypotheses.append(dict(selector=selector,order_budget=budget,treatment=treatment,control=control))
                differences.append(returns[f'{selector}__{treatment}__orders{budget}']-returns[f'{selector}__{control}__orders{budget}'])
    statistics=[]
    for block in spec['bootstrap']['blocks']:
        result=family_bootstrap(np.column_stack(differences),block=block,draws=4000,seed=81)
        for i,hypothesis in enumerate(hypotheses):
            statistics.append(dict(hypothesis,block=block,**{k:float(result[k][i]) for k in
                              ['mean','standard_error','adjusted_p','simultaneous_lower95']}))
    pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
    pd.DataFrame(statistics).to_csv(DEST/'bootstrap.csv',index=False)
    atomic_json(DEST/'independent_audit.json',audits)
    atomic_json(DEST/'summary.json',{'status':'development_only','portfolios':len(rows),'hypotheses':len(hypotheses),
                                   'covered_signal_sessions':int(available.sum()),'statistics':statistics})
    print(json.dumps({'complete':True,'portfolios':len(rows),'hypotheses':len(hypotheses)}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);args=ap.parse_args()
    if args.action=='freeze':freeze()
    else:run()

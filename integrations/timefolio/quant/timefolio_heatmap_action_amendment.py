"""Corporate-action ledger correction of every fixed v4 portfolio comparison.

Mandatory source corrections are applied equally to all forecasts and controls;
no event treatment is chosen by its effect on strategy returns.
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
from quant.timefolio_heatmap_execution_amendment import (
    DEST as PRIOR, corrected_panel, prediction_receipt, BUDGETS,
)
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_replay import INITIAL_NAV
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores, market_regimes, calendar_blocks, family_bootstrap

DEST=SOURCE.with_name('20260928_action_amendment_v5')
REGISTRY=SOURCE.with_name('20260928_planning_probe_v1')/'verified_actions_followup.json'


def amend_actions(p,ix,actions):
    panel,release=corrected_panel(p,ix)
    for action in actions:
        i=ix['codes'].index(action['code']);d=ix['dates'].index(action['ex_date'])
        panel['split'][i,d]=action['quantity_ratio']
        if action['quantity_ratio']>1:
            release[(action['code'],action['ex_date'])]=action['listing_date']
    return panel,release


def release_audit(p,ix,result,release):
    """Independent per-security entitlement ledger checks sellable quantities."""
    grouped={};errors=[];dates={d:i for i,d in enumerate(ix['dates'])}
    for t in result['trades']:grouped.setdefault((t['code'],t['date']),[]).append(t)
    for code in {key[0] for key in release}:
        i=ix['codes'].index(code);qty=0;pending=[]
        for row in result['daily']:
            d=dates[row['date']];before=qty;ratio=p['split'][i,d]
            qty=int(np.floor(qty*ratio+1e-7))
            pending=[(date,int(np.floor(q*min(ratio,1)))) for date,q in pending]
            listing=release.get((code,row['date']))
            if listing and qty>before:pending.append((listing,qty-before))
            pending=[(date,q) for date,q in pending if date>row['date']]
            locked=sum(q for _,q in pending)
            for t in grouped.get((code,row['date']),[]):
                if t['side']=='buy':qty+=t['qty']
                else:
                    if t['qty']>qty-locked:errors.append({'date':row['date'],'code':code,'kind':'sold_announced_locked_entitlement'})
                    qty-=t['qty']
    return errors


def freeze():
    actions=json.loads(REGISTRY.read_text())['actions'];prior=json.loads((PRIOR/'protocol.json').read_text())
    files=[Path(__file__),REGISTRY,PRIOR/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['execution_amendment','planned_replay','planned_audit','planning_probe','walkforward_eval','walkforward','study','replay','features','data']]
    spec={'prior':prior,'actions':actions,'status':'development_only; mandatory ledger correction, not a new strategy search',
          'scope':'same 384 v4 portfolios and 432 hypotheses; inputs/forecasts unchanged; source-based action ratios and release dates only',
          'remaining_limitations':'Other inferred corporate events, cash dividends, historical order books, model features/labels and original data-universe limitations remain.',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Action protocol changed')
    if not path.exists():
        atomic_json(path,spec);frozen=DEST/'frozen_source';frozen.mkdir(exist_ok=True)
        for n,p in enumerate(files):shutil.copy2(p,frozen/(f'{n:02d}_'+p.name))
    return spec


def run():
    spec=freeze();base=spec['prior'];model_spec=base['model_protocol'];prediction_receipt(MODEL_ROOT,DEST)
    p,ix,ci,di=context(SOURCE);p,release=amend_actions(p,ix,spec['actions'])
    scores,choices,neural=load_scores(MODEL_ROOT,p,ix,ci,di,model_spec);schedules,_=market_regimes(p)
    atomic_json(DEST/'online_choices.json',choices)
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True);rows=[];returns={};audits={}
    for policy in model_spec['policies']:
        panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],policy['sector_cap']);schedule=schedules[policy['regime']]
        for budget in BUDGETS:
            for name,matrix in scores.items():
                key=f"{name}__{policy['id']}__orders{budget}";path=folder/(key+'.json')
                if path.exists():r=json.loads(path.read_text())
                else:
                    r=replay(panel,ix,matrix,model_spec['evaluation_start'],model_spec['evaluation_end'],
                             top_n=policy['top_n'],weight=policy['stock_weight'],gross_schedule=schedule,
                             max_orders=budget,return_trades=True,planning_price='open',action_release_dates=release)
                    atomic_json(path,r)
                check=audit_fills(panel,ix,r);check.update(additional_checks(panel,ix,r,schedule,max_orders=budget))
                check['announced_action_errors']=release_audit(panel,ix,r,release)
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['maximum_nav_reconstruction_error_krw']>.01):
                    atomic_json(DEST/'failed_audit.json',{'id':key,'audit':check});raise AssertionError('Action ledger audit failed')
                audits[key]=check;nav=np.array([x['nav'] for x in r['daily']]);returns[key]=nav/np.r_[INITIAL_NAV,nav[:-1]]-1
                quarters=calendar_blocks(r['daily'])
                rows.append(dict(id=key,model=name,policy=policy['id'],order_budget=budget,
                                 **r['metrics'],**quarters,positive_blocks=sum(x>0 for x in quarters.values())))
            pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
            print(json.dumps({'policy':policy['id'],'orders':budget,'portfolios':len(rows)}),flush=True)
    atomic_json(DEST/'independent_audit.json',audits)
    family=[];differences=[]
    for policy in model_spec['policies']:
        for budget in BUDGETS:
            suffix=f"__{policy['id']}__orders{budget}";control=returns['online_nonimage'+suffix]
            for name in neural:
                key=name+suffix
                for comparator,baseline in [('cash',0.),('nonimage',control)]:
                    family.append((key,comparator));differences.append(returns[key]-baseline)
    if len(family)!=432:raise AssertionError('Unexpected comparison family')
    matrix=np.column_stack(differences);stats=[]
    for block in base['bootstrap']['blocks']:
        result=family_bootstrap(matrix,block=block,draws=4000,seed=57)
        for i,(key,comparator) in enumerate(family):
            stats.append(dict(id=key,comparator=comparator,block=block,**{k:float(result[k][i]) for k in
                         ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    statistics=pd.DataFrame(stats);statistics.to_csv(DEST/'bootstrap.csv',index=False);candidates=[]
    for row in rows:
        if row['model'] not in neural:continue
        inference=statistics[statistics.id==row['id']]
        gate=row['return']>0 and row['positive_blocks']>=2 and row['mdd']>-.20 and not row['four_week_turnover_stop']
        if gate and (inference.adjusted_p<.05).all() and (inference.simultaneous_lower95>0).all():candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json',{'status':'development_only','portfolios':len(rows),'family_hypotheses':len(family),
                                              'days':len(matrix),'candidate_gate_passed':candidates,'independent_confirmation':False})
    prior=pd.read_csv(PRIOR/'portfolio_summary.csv');current=pd.DataFrame(rows)
    compare=current[['id','return','mdd','low_turnover_weeks','unreleased_rights_positions']].merge(
        prior[['id','return','mdd','low_turnover_weeks','unreleased_rights_positions']],on='id',suffixes=('_verified','_v4'))
    compare['return_change']=compare.return_verified-compare.return_v4
    compare.to_csv(DEST/'action_effect.csv',index=False)
    print(json.dumps({'complete':True,'portfolios':len(rows),'candidates':candidates}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:run()

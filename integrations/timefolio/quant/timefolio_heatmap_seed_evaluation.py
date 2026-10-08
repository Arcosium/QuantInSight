"""Seed robustness with a joint comparison family, never best-seed selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_action_amendment import DEST as BASE_ROOT, amend_actions, release_audit
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_replay import INITIAL_NAV
from quant.timefolio_heatmap_seed_replication import DEST as SEED_ROOT, MODEL, SEEDS
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT, SOURCE, monthly_folds
from quant.timefolio_heatmap_walkforward_eval import market_regimes, per_date_rank, calendar_blocks, family_bootstrap

DEST=SOURCE.with_name('20260928_seed_evaluation_v1')


def merge_months(folder,di,dates):
    out=np.full(len(di),np.nan,np.float32)
    for fold in monthly_folds(dates):
        path=folder/(fold['month']+'.json');row=json.loads(path.read_text())
        if row['last_refit_label']>=row['first_execution']:raise AssertionError('Future label')
        values=np.load(path.with_suffix('.pred.npy'))
        mask=(di>=fold['start']-1)&(di<fold['end']-1)
        if not np.array_equal(np.isfinite(values),mask):raise AssertionError('Forecast date coverage differs')
        out[mask]=values[mask]
    return out


def daily_returns(result):
    nav=np.array([x['nav'] for x in result['daily']])
    return nav/np.r_[INITIAL_NAV,nav[:-1]]-1


def freeze():
    base=json.loads((BASE_ROOT/'protocol.json').read_text());replica=json.loads((SEED_ROOT/'protocol.json').read_text())
    files=[Path(__file__),BASE_ROOT/'protocol.json',SEED_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['action_amendment','execution_amendment','planned_replay','planned_audit','seed_replication',
               'walkforward_eval','walkforward','study','replay','features','data']]
    spec={'base':base,'replication':replica,
          'joint_family':'v5 432 hypotheses plus 72 new seed29/43/equal-rank-ensemble hypotheses = 504',
          'reuse':'seed17 replay must reproduce its v5 NAV path exactly; every original comparison remains in the bootstrap family',
          'ensemble':'equal mean of same-date percentile ranks for seeds17,29,43; fixed before new-seed outcomes',
          'robust_candidate_gate':'ensemble passes joint-family gate; all three individual seeds have positive return, MDD above -20%, at least two positive quarters and fewer than four turnover violations',
          'bootstrap':{'blocks':[5,10],'draws':4000,'seed':57},
          'interpretation':'exploratory seed robustness on reused history; not independent confirmation',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Seed evaluation specification changed')
    if not path.exists():
        atomic_json(path,spec);frozen=DEST/'frozen_source';frozen.mkdir(exist_ok=True)
        for n,p in enumerate(files):shutil.copy2(p,frozen/(f'{n:02d}_'+p.name))
    return spec


def run():
    spec=freeze();base=spec['base'];p,ix,ci,di=context(SOURCE);p,release=amend_actions(p,ix,base['actions'])
    regimes,_=market_regimes(p);model_spec=base['prior']['model_protocol'];budgets=base['prior']['order_budgets']
    values={17:merge_months(MODEL_ROOT/'models'/MODEL,di,ix['dates'])}
    values.update({seed:merge_months(SEED_ROOT/'models'/f'seed{seed}',di,ix['dates']) for seed in SEEDS})
    forecast_paths=list((SEED_ROOT/'models').glob('*/*.pred.npy'))
    atomic_json(DEST/'forecast_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in forecast_paths})
    ensemble=np.mean([per_date_rank(values[s],di) for s in [17,*SEEDS]],axis=0)
    matrices={f'seed{s}':score_matrix(v,ci,di,p['close'].shape) for s,v in values.items()}
    matrices['ensemble']=score_matrix(ensemble,ci,di,p['close'].shape)
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True);rows=[];returns={};audits={}
    for policy in model_spec['policies']:
        panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],policy['sector_cap']);schedule=regimes[policy['regime']]
        for budget in budgets:
            for name,matrix in matrices.items():
                key=f"{name}__{policy['id']}__orders{budget}";path=folder/(key+'.json')
                if path.exists():result=json.loads(path.read_text())
                else:
                    result=replay(panel,ix,matrix,model_spec['evaluation_start'],model_spec['evaluation_end'],
                                  top_n=policy['top_n'],weight=policy['stock_weight'],max_orders=budget,gross_schedule=schedule,
                                  return_trades=True,planning_price='open',action_release_dates=release)
                    atomic_json(path,result)
                if name=='seed17':
                    prior=json.loads((BASE_ROOT/'portfolios'/f"{MODEL}__{policy['id']}__orders{budget}.json").read_text())
                    if prior['daily']!=result['daily']:raise AssertionError('Seed17 does not reproduce frozen v5')
                check=audit_fills(panel,ix,result);check.update(additional_checks(panel,ix,result,schedule,max_orders=budget))
                check['announced_action_errors']=release_audit(panel,ix,result,release)
                if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                    or check['maximum_nav_reconstruction_error_krw']>.01):
                    atomic_json(DEST/'failed_audit.json',{'id':key,'audit':check});raise AssertionError('Seed audit failed')
                audits[key]=check;returns[key]=daily_returns(result);quarters=calendar_blocks(result['daily'])
                rows.append(dict(id=key,model=name,policy=policy['id'],order_budget=budget,
                                 **result['metrics'],**quarters,positive_blocks=sum(x>0 for x in quarters.values())))
            print(json.dumps({'policy':policy['id'],'orders':budget,'portfolios':len(rows)}),flush=True)
    atomic_json(DEST/'independent_audit.json',audits);frame=pd.DataFrame(rows);frame.to_csv(DEST/'portfolio_summary.csv',index=False)
    original=pd.read_csv(BASE_ROOT/'bootstrap.csv');family=[];differences=[];cache={}
    def original_returns(key):
        if key not in cache:cache[key]=daily_returns(json.loads((BASE_ROOT/'portfolios'/(key+'.json')).read_text()))
        return cache[key]
    for key,comp in original[['id','comparator']].drop_duplicates().itertuples(index=False,name=None):
        suffix=key.split('__',1)[1]
        baseline=0. if comp=='cash' else original_returns('online_nonimage__'+suffix)
        family.append((key,comp,'original'));differences.append(original_returns(key)-baseline)
    if len(family)!=432:raise AssertionError('Original family is incomplete')
    for policy in model_spec['policies']:
        for budget in budgets:
            suffix=f"__{policy['id']}__orders{budget}";control=original_returns('online_nonimage'+suffix)
            for name in ['seed29','seed43','ensemble']:
                key=name+suffix
                for comp,baseline in [('cash',0.),('nonimage',control)]:
                    family.append((key,comp,'replication'));differences.append(returns[key]-baseline)
    if len(family)!=504:raise AssertionError('Unexpected joint family')
    matrix=np.column_stack(differences);statistics=[]
    for block in spec['bootstrap']['blocks']:
        result=family_bootstrap(matrix,block=block,draws=4000,seed=57)
        for i,(key,comp,origin) in enumerate(family):
            statistics.append(dict(id=key,comparator=comp,origin=origin,block=block,
                                   **{k:float(result[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    stats=pd.DataFrame(statistics);stats.to_csv(DEST/'joint_bootstrap.csv',index=False);candidates=[]
    for row in rows:
        if row['model']!='ensemble':continue
        infer=stats[(stats.id==row['id'])&(stats.origin=='replication')]
        members=frame[(frame.policy==row['policy'])&(frame.order_budget==row['order_budget'])]
        stable=((members['return']>0)&(members.mdd>-.20)&(members.positive_blocks>=2)&(~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p<.05).all() and (infer.simultaneous_lower95>0).all():candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json',{'status':'development_only','portfolios':len(rows),'joint_family_hypotheses':len(family),
                                              'robust_candidate_gate_passed':candidates,'independent_confirmation':False})
    print(json.dumps({'complete':True,'robust_candidates':candidates}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:run()

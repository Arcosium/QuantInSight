"""Causal model selection using previously realised, costed shadow portfolios.

This is a new exploratory rule motivated by the failure of inner-IC selection.
It reuses development history and cannot provide independent confirmation.
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
from quant.timefolio_heatmap_action_amendment import DEST as BASE_ROOT,amend_actions,release_audit
from quant.timefolio_heatmap_seed_evaluation import DEST as SEED_EVAL,daily_returns
from quant.timefolio_heatmap_planned_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks
from quant.timefolio_heatmap_study import context,score_matrix
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE,monthly_folds
from quant.timefolio_heatmap_walkforward_eval import load_scores,per_date_rank,market_regimes,calendar_blocks,family_bootstrap

DEST=SOURCE.with_name('20260928_portfolio_selector_v1')
CASES=[{'id':f'past{window}_top{count}','lookback':window,'count':count} for window in [20,60] for count in [1,3]]


def select_past(pool,returns,dates,origin,lookback,count,minimum_days=20):
    """The origin day's return and every later return are excluded."""
    dates=np.asarray(dates)
    if dates.ndim!=1 or np.any(dates[1:]<=dates[:-1]):raise ValueError('Unique ordered shadow dates required')
    use=np.flatnonzero(dates<origin)[-lookback:]
    if len(use)<minimum_days:
        return list(pool),{'observations':len(use),'fallback':'equal rank average of the complete pool','last_observation':str(dates[use[-1]]) if len(use) else None}
    utilities={}
    for name in pool:
        r=np.asarray(returns[name],float)
        if r.shape!=dates.shape or not np.isfinite(r[use]).all():raise ValueError('Invalid past shadow returns')
        utilities[name]=float(r[use].mean()/max(r[use].std(ddof=1),1e-6)*np.sqrt(252))
    selected=sorted(pool,key=lambda n:(-utilities[n],n))[:count]
    return selected,{'observations':len(use),'last_observation':str(dates[use[-1]]),
                     'past_annualised_sharpe':{name:utilities[name] for name in selected}}


def freeze():
    base=json.loads((BASE_ROOT/'protocol.json').read_text())
    files=[Path(__file__),BASE_ROOT/'protocol.json',SEED_EVAL/'protocol.json',BASE_ROOT/'portfolio_summary.csv']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['action_amendment','planned_replay','planned_audit','walkforward_eval','walkforward','study','replay','features','data','seed_evaluation']]
    spec={'base':base,'cases':CASES,'minimum_history':20,
          'selection':'each month before its first execution; trailing 20/60 realised sessions of the corresponding v5 policy/order-budget shadow portfolio; highest net daily-return Sharpe',
          'pools':'16 individual CNN/split/TCN versus six individual MLP/GBM; original adaptive selectors excluded',
          'warmup':'fewer than 20 prior observations: equal within-date rank average of the complete respective pool',
          'execution':'one continuous actual portfolio for each selector; switching signals never switches NAV or silently transfers holdings',
          'comparison_family':'504 original and seed hypotheses plus 48 new neural selectors x cash/matched nonimage = 600',
          'inference':'one joint centred circular-block max-t family; blocks5/10, draws4000, seed57; no family reset',
          'status':'exploratory development; devised after previous full-batch outcomes; independent confirmation absent',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Portfolio selector protocol changed')
    if not path.exists():
        atomic_json(path,spec);folder=DEST/'frozen_source';folder.mkdir(exist_ok=True)
        for n,p in enumerate(files):shutil.copy2(p,folder/(f'{n:02d}_'+p.name))
    return spec


def run():
    spec=freeze();base=spec['base'];model_spec=base['prior']['model_protocol']
    p,ix,ci,di=context(SOURCE);p,release=amend_actions(p,ix,base['actions']);regimes,_=market_regimes(p)
    scores,_,_=load_scores(MODEL_ROOT,p,ix,ci,di,model_spec)
    pools={'neural':[c['id'] for c in model_spec['configs'] if c['architecture'] in ['cnn','split','tcn']],
           'nonimage':[c['id'] for c in model_spec['configs'] if c['architecture'] in ['mlp','gbm']]}
    names=pools['neural']+pools['nonimage']
    ranks={name:score_matrix(per_date_rank(scores[name][ci,di],di),ci,di,p['close'].shape) for name in names}
    del scores
    output=DEST/'portfolios';output.mkdir(exist_ok=True);rows=[];choices=[];audits={};shadow_hashes={}
    for policy in model_spec['policies']:
        panel=dict(p);panel['sector_cap']=np.minimum(p['sector_cap'],policy['sector_cap']);schedule=regimes[policy['regime']]
        for budget in base['prior']['order_budgets']:
            shadow={};shadow_dates=None
            for name in names:
                path=BASE_ROOT/'portfolios'/f"{name}__{policy['id']}__orders{budget}.json"
                data=path.read_bytes();r=json.loads(data);shadow_hashes[str(path)]=hashlib.sha256(data).hexdigest()
                dates=[x['date'] for x in r['daily']]
                if shadow_dates is not None and dates!=shadow_dates:raise AssertionError('Shadow calendars differ')
                shadow_dates=dates;shadow[name]=daily_returns(r)
            for case in CASES:
                for kind,pool in pools.items():
                    name=case['id']+'_'+kind;key=f"{name}__{policy['id']}__orders{budget}"
                    matrix=np.full(p['close'].shape,np.nan,np.float32)
                    for fold in monthly_folds(ix['dates']):
                        origin=ix['dates'][fold['start']]
                        selected,info=select_past(pool,shadow,shadow_dates,origin,case['lookback'],case['count'])
                        first,last=fold['start']-1,fold['end']-1
                        matrix[:,first:last]=np.mean([ranks[n][:,first:last] for n in selected],axis=0)
                        choices.append(dict(id=key,month=fold['month'],first_execution=origin,chosen=selected,**info))
                    path=output/(key+'.json')
                    if path.exists():r=json.loads(path.read_text())
                    else:
                        r=replay(panel,ix,matrix,model_spec['evaluation_start'],model_spec['evaluation_end'],
                                 top_n=policy['top_n'],weight=policy['stock_weight'],gross_schedule=schedule,
                                 max_orders=budget,return_trades=True,planning_price='open',action_release_dates=release)
                        atomic_json(path,r)
                    check=audit_fills(panel,ix,r);check.update(additional_checks(panel,ix,r,schedule,max_orders=budget))
                    check['announced_action_errors']=release_audit(panel,ix,r,release)
                    if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                        or check['maximum_nav_reconstruction_error_krw']>.01):raise AssertionError('Portfolio-selector ledger audit failed')
                    audits[key]=check;quarters=calendar_blocks(r['daily'])
                    rows.append(dict(id=key,selector=case['id'],kind=kind,policy=policy['id'],order_budget=budget,
                                     **r['metrics'],**quarters,positive_blocks=sum(v>0 for v in quarters.values())))
            print(json.dumps({'policy':policy['id'],'orders':budget,'portfolios':len(rows)}),flush=True)
    pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
    atomic_json(DEST/'choices.json',choices);atomic_json(DEST/'independent_audit.json',audits)
    atomic_json(DEST/'shadow_hashes.json',shadow_hashes)
    print(json.dumps({'portfolio_stage_complete':True,'portfolios':len(rows)}),flush=True)


def evaluate():
    freeze();prior=pd.read_csv(SEED_EVAL/'joint_bootstrap.csv');new=pd.read_csv(DEST/'portfolio_summary.csv')
    family=[];differences=[];cache={}
    def returns(root,key):
        name=(str(root),key)
        if name not in cache:cache[name]=daily_returns(json.loads((root/'portfolios'/(key+'.json')).read_text()))
        return cache[name]
    for key,comp,origin in prior[['id','comparator','origin']].drop_duplicates().itertuples(index=False,name=None):
        source=BASE_ROOT if origin=='original' else SEED_EVAL;suffix=key.split('__',1)[1]
        baseline=0. if comp=='cash' else returns(BASE_ROOT,'online_nonimage__'+suffix)
        family.append((key,comp,origin));differences.append(returns(source,key)-baseline)
    if len(family)!=504:raise AssertionError('Prior seed family incomplete')
    for row in new[new.kind=='neural'].to_dict('records'):
        key=row['id'];control=key.replace('_neural__','_nonimage__')
        for comp,baseline in [('cash',0.),('nonimage',returns(DEST,control))]:
            family.append((key,comp,'portfolio_selector'));differences.append(returns(DEST,key)-baseline)
    if len(family)!=600:raise AssertionError('Joint family size changed')
    matrix=np.column_stack(differences);stats=[]
    for block in [5,10]:
        result=family_bootstrap(matrix,block=block,draws=4000,seed=57)
        for i,(key,comp,origin) in enumerate(family):
            stats.append(dict(id=key,comparator=comp,origin=origin,block=block,
                              **{k:float(result[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    statistics=pd.DataFrame(stats);statistics.to_csv(DEST/'joint_bootstrap.csv',index=False);candidates=[]
    for row in new[new.kind=='neural'].to_dict('records'):
        infer=statistics[(statistics.id==row['id'])&(statistics.origin=='portfolio_selector')]
        gate=row['return']>0 and row['mdd']>-.20 and row['positive_blocks']>=2 and not row['four_week_turnover_stop']
        if gate and (infer.adjusted_p<.05).all() and (infer.simultaneous_lower95>0).all():candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json',{'status':'development_only','portfolios':len(new),'joint_hypotheses':len(family),
                                              'candidate_gate_passed':candidates,'independent_confirmation':False})
    print(json.dumps({'complete':True,'candidates':candidates}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run','stats']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='run':run()
    else:evaluate()

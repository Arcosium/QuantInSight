"""Explore target count and a pre-trade adjustment band under tight order budgets.

All outcomes are development evidence. The old hypothesis families remain in
one joint test; no chosen parameter or seed is silently discarded from counting.
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
from quant.timefolio_heatmap_seed_evaluation import DEST as SEED_ROOT,daily_returns
from quant.timefolio_heatmap_portfolio_selector import DEST as SELECTOR_ROOT
from quant.timefolio_heatmap_banded_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills,additional_checks
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE
from quant.timefolio_heatmap_walkforward_eval import load_scores,calendar_blocks,family_bootstrap

DEST=SOURCE.with_name('20260928_allocation_probe_v1')
NEURAL=['monthly_binary','cnn_sector_h5','online_neural3']
MODELS=NEURAL+['online_nonimage','lowvol','momentum5']
CASES=[{'id':f'n{n}__orders{orders}__band{bp}bp','top_n':n,'max_orders':orders,'rebalance_band':bp/10000}
       for n in [4,8,12] for orders in [3,10] for bp in [0,5]]


def freeze():
    base=json.loads((BASE_ROOT/'protocol.json').read_text())
    files=[Path(__file__),BASE_ROOT/'protocol.json',SELECTOR_ROOT/'protocol.json']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['banded_replay','action_amendment','planned_audit','walkforward_eval','walkforward',
               'study','replay','features','data','seed_evaluation','portfolio_selector']]
    spec={'base':base,'models':MODELS,'neural':NEURAL,'cases':CASES,
          'fixed_policy':'stock5%, sector min(statutory,20%), gross ceiling80%, rebalance5; no exposure regime',
          'band':'ignore trim/top-up requests below 5bp of planning NAV only for continuing eligible positions without prior-close or planning-price risk breaches; entries and full exits retained',
          'selection_context':'after v5 and macro outcomes; two early promising individual CNNs and the pre-existing causal three-neural average; common nonimage selector plus simple controls',
          'joint_family':'600 prior hypotheses plus 36 new neural paths x cash/matched nonimage = 672',
          'statistics':'same centred circular-block max-t, blocks5/10, draws4000, seed57',
          'status':'adaptive development, not independent confirmation; remaining corporate/dividend/quote/universe limitations retained',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Allocation protocol changed')
    if not path.exists():
        atomic_json(path,spec);folder=DEST/'frozen_source';folder.mkdir(exist_ok=True)
        for i,p in enumerate(files):shutil.copy2(p,folder/(f'{i:02d}_'+p.name))
    return spec


def run():
    spec=freeze();base=spec['base'];model_spec=base['prior']['model_protocol']
    p,ix,ci,di=context(SOURCE);p,release=amend_actions(p,ix,base['actions'])
    p['sector_cap']=np.minimum(p['sector_cap'],.20)
    scores,_,_=load_scores(MODEL_ROOT,p,ix,ci,di,model_spec)
    # A real-history regression check guards the fork before any new comparisons.
    original=json.loads((BASE_ROOT/'portfolios'/'cnn_sector_h5__sector20__orders3.json').read_text())
    check=replay(p,ix,scores['cnn_sector_h5'],model_spec['evaluation_start'],model_spec['evaluation_end'],
                 top_n=20,weight=.05,max_orders=3,return_trades=True,action_release_dates=release)
    if check!=original:raise AssertionError('Zero-band real-history replay changed')
    atomic_json(DEST/'zero_band_regression.json',{'case':'cnn_sector_h5__sector20__orders3','all_fields_equal':True})
    folder=DEST/'portfolios';folder.mkdir(exist_ok=True);rows=[];audits={}
    for case in CASES:
        for name in MODELS:
            key=name+'__'+case['id'];path=folder/(key+'.json')
            if path.exists():result=json.loads(path.read_text())
            else:
                args={k:v for k,v in case.items() if k!='id'}
                result=replay(p,ix,scores[name],model_spec['evaluation_start'],model_spec['evaluation_end'],
                              weight=.05,return_trades=True,planning_price='open',action_release_dates=release,**args)
                atomic_json(path,result)
            check=audit_fills(p,ix,result);check.update(additional_checks(p,ix,result,None,max_orders=case['max_orders']))
            check['announced_action_errors']=release_audit(p,ix,result,release)
            if (check['post_buy_limit_violations'] or check['additional_errors'] or check['announced_action_errors']
                or check['maximum_nav_reconstruction_error_krw']>.01):raise AssertionError('Allocation audit failed')
            audits[key]=check;quarters=calendar_blocks(result['daily'])
            rows.append(dict(id=key,model=name,case=case['id'],top_n=case['top_n'],order_budget=case['max_orders'],
                             band=case['rebalance_band'],**result['metrics'],**quarters,positive_blocks=sum(x>0 for x in quarters.values())))
        print(json.dumps({'case':case['id'],'portfolios':len(rows)}),flush=True)
    pd.DataFrame(rows).to_csv(DEST/'portfolio_summary.csv',index=False)
    atomic_json(DEST/'independent_audit.json',audits)
    print(json.dumps({'portfolio_stage_complete':True,'portfolios':len(rows)}),flush=True)


def evaluate():
    freeze();prior=pd.read_csv(SELECTOR_ROOT/'joint_bootstrap.csv');new=pd.read_csv(DEST/'portfolio_summary.csv')
    family=[];differences=[];cache={}
    def returns(root,key):
        name=(str(root),key)
        if name not in cache:cache[name]=daily_returns(json.loads((root/'portfolios'/(key+'.json')).read_text()))
        return cache[name]
    roots={'original':BASE_ROOT,'replication':SEED_ROOT,'portfolio_selector':SELECTOR_ROOT}
    for key,comp,origin in prior[['id','comparator','origin']].drop_duplicates().itertuples(index=False,name=None):
        source=roots[origin]
        if origin=='portfolio_selector':control_root=source;control=key.replace('_neural__','_nonimage__')
        else:control_root=BASE_ROOT;control='online_nonimage__'+key.split('__',1)[1]
        baseline=0. if comp=='cash' else returns(control_root,control)
        family.append((key,comp,origin));differences.append(returns(source,key)-baseline)
    if len(family)!=600:raise AssertionError('Prior family incomplete')
    for row in new[new.model.isin(NEURAL)].to_dict('records'):
        for comp,baseline in [('cash',0.),('nonimage',returns(DEST,'online_nonimage__'+row['case']))]:
            family.append((row['id'],comp,'allocation'));differences.append(returns(DEST,row['id'])-baseline)
    if len(family)!=672:raise AssertionError('Joint family changed')
    matrix=np.column_stack(differences);stats=[]
    for block in [5,10]:
        result=family_bootstrap(matrix,block=block,draws=4000,seed=57)
        for i,(key,comp,origin) in enumerate(family):
            stats.append(dict(id=key,comparator=comp,origin=origin,block=block,
                              **{k:float(result[k][i]) for k in ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    statistics=pd.DataFrame(stats);statistics.to_csv(DEST/'joint_bootstrap.csv',index=False);candidates=[]
    for row in new[new.model.isin(NEURAL)].to_dict('records'):
        infer=statistics[(statistics.id==row['id'])&(statistics.origin=='allocation')]
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

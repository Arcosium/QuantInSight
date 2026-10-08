"""Seed replication and fixed rank averaging of the H10 pairwise hypothesis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_rank_training import DEST as TRAIN_PRIOR, CONFIGS as BASE_CONFIGS, train_daywise
from quant.timefolio_heatmap_rank_evaluation import DEST as RANK_ROOT, CASES
from quant.timefolio_heatmap_h10_control import DEST as PRIOR, retained_family
from quant.timefolio_heatmap_action_amendment import amend_actions, release_audit
from quant.timefolio_heatmap_seed_evaluation import merge_months
from quant.timefolio_heatmap_consensus import rank_consensus
from quant.timefolio_heatmap_snapshot import snapshot_scores
from quant.timefolio_heatmap_stock_limits import historical_stock_caps, audit_pre_july_hynix
from quant.timefolio_heatmap_week_boundaries import assess_weeks
from quant.timefolio_heatmap_dated_replay import replay
from quant.timefolio_heatmap_planned_audit import audit_fills, additional_checks
from quant.timefolio_heatmap_study import context, score_matrix
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT, AlternativeNet, continuous_targets, monthly_folds, fold_masks, predict
from quant.timefolio_heatmap_walkforward_eval import calendar_blocks, family_bootstrap

DEST = SOURCE.with_name('20260929_rank_seeds_v1')
SEEDS = [29, 43]
CONFIGS = [dict(c, id=c['id']+f'_seed{seed}', seed=seed)
           for c in BASE_CONFIGS if c['objective']=='pairwise' and c['horizon']==10 for seed in SEEDS]


def freeze():
    parent = json.loads((TRAIN_PRIOR/'protocol.json').read_text())
    for path,digest in parent['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest: raise AssertionError('Training parent changed')
    files = [Path(__file__), TRAIN_PRIOR/'protocol.json', PRIOR/'protocol.json', PRIOR/'joint_bootstrap.csv',
             TRAIN_PRIOR/'labels_h10.npy', MODEL_ROOT/'images_raw.npy', SOURCE/'panel.npz', SOURCE/'panel_index.json', SOURCE/'samples.npz']
    for arch in ['cnn','mlp']:
        folder = TRAIN_PRIOR/'models'/(arch+'_pairwise_h10')
        files += sorted(folder.glob('*.json'))+sorted(folder.glob('*.pred.npy'))
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['rank_training','rank_evaluation','h10_control','concentration','consensus','snapshot','stock_limits',
               'week_boundaries','dated_replay','action_amendment','execution_amendment','planned_audit',
               'seed_evaluation','walkforward_eval','walkforward','study','features','replay','data']]
    spec = {'configs':CONFIGS, 'training_parent':parent, 'cases':CASES, 'existing_seed':17, 'new_seeds':SEEDS,
            'training':'same H10 corrected labels, raw images, architecture, date batches, optimiser and past-validation selection; only seed/id change; all36 new monthly fits retained',
            'ensemble':'equal mean of eligible same-signal-date percentile ranks from seeds17/29/43; no outcome weighting or best-seed selection; shared coverage required; identical method for CNN/MLP',
            'execution':'same8cases as rank evaluation; target5%, gross80%, sector min(statutory,20%), small-cap30%; dated stock caps and completed-holiday-week turnover',
            'baseline':'both seed17 models replayed in8cases and all fields must equal original rank evaluation; those16 are regression checks, not additional hypotheses',
            'family':'3224 previous full-period hypotheses plus3newCNN paths(seed29/43/ensemble) x8cases x3comparators =3296',
            'comparators':'cash, matching-seed-or-ensemble MLP in the same case, and original same-case online_nonimage',
            'bootstrap':{'blocks':[5,10], 'draws':4000, 'seed':57, 'alpha':.025},
            'candidate_gate':'only fixed ensemble can qualify; its adjustedp<0.025 and simultaneous lower95>0 for all3comparators/bothblocks; ensemble and each of3individualCNN seeds must have positive return, MDD>-20%, >=2positivequarters and fewer than4turnover failures; subsequent stress and independent confirmation remain required',
            'context':'adaptive development after H10 loss comparison; seed17 alone showed directional advantage but failed significance; old data/order-book/action/universe limits remain; no reserved fresh outcome use',
            'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec: raise RuntimeError('Rank seed protocol changed')
    if not path.exists():
        atomic_json(path,spec); folder=DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i,p in enumerate(files):
            if p.suffix in ['.py','.json']: shutil.copy2(p,folder/(f'{i:02d}_'+p.name))
    return spec


def train():
    spec=freeze(); torch.set_num_threads(4)
    p,ix,ci,di=context(SOURCE); p,_=amend_actions(p,ix,spec['training_parent']['actions']); p['factor']=np.cumprod(p['split'],axis=1)
    y=continuous_targets(p,ci,di,10,'sector')
    if not np.array_equal(y,np.load(TRAIN_PRIOR/'labels_h10.npy'),equal_nan=True): raise AssertionError('Replica labels differ')
    dates=np.asarray(ix['dates']); valid=di+1<len(dates); allowed=np.zeros(len(ci),bool)
    allowed[valid]=p['trade_allowed'][ci[valid],di[valid]+1].astype(bool)
    x=torch.from_numpy(np.array(np.load(MODEL_ROOT/'images_raw.npy',mmap_mode='r')[:,None],copy=True)); matches=[]
    for cfg in CONFIGS:
        folder=DEST/'models'/cfg['id']; folder.mkdir(parents=True,exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path=folder/(fold['month']+'.json'); parent_id=cfg['architecture']+'_pairwise_h10'
            reference=json.loads((TRAIN_PRIOR/'models'/parent_id/path.name).read_text())
            if {k:v for k,v in cfg.items() if k not in ['id','seed']}!={k:v for k,v in reference['config'].items() if k not in ['id','seed']}:
                raise AssertionError('Replica differs beyond seed')
            tr,va,rf,pr,_=fold_masks(di,y,allowed,10,fold['start'],fold['end'])
            if int(tr.sum())!=reference['fit']['training_rows'] or int(va.sum())!=reference['fit']['validation_rows'] or int(rf.sum())!=reference['refit_rows']:
                raise AssertionError('Replica data differ')
            matches.append({'model':cfg['id'],'month':fold['month'],'same_data_and_config_except_seed':True})
            if path.exists():
                if not path.with_suffix('.pt').exists() or not path.with_suffix('.pred.npy').exists(): raise RuntimeError('Incomplete seed checkpoint')
                continue
            started=time.monotonic(); model,fit=train_daywise(x,y,tr,va,cfg,di); del model
            model,_=train_daywise(x,y,rf,np.zeros(len(y),bool),cfg,di,epochs=fit['best_epoch'])
            values=predict(model,x,np.flatnonzero(pr))
            if not np.isfinite(values).all(): raise AssertionError('Non-finite replica scores')
            out=np.full(len(ci),np.nan,np.float32); out[pr]=values
            torch.save({'config':cfg,'state_dict':model.state_dict()},path.with_suffix('.pt')); np.save(path.with_suffix('.pred.npy'),out)
            row={'config':cfg,'fold':fold,'fit':fit,'refit_rows':int(rf.sum()),
                 'last_inner_train_label':str(max(dates[di[tr]+10])), 'first_inner_validation_signal':str(min(dates[di[va]])),
                 'last_refit_label':str(max(dates[di[rf]+10])), 'first_execution':ix['dates'][fold['start']],
                 'last_execution':ix['dates'][fold['end']-1], 'seconds':round(time.monotonic()-started,2)}
            for key in ['last_inner_train_label','first_inner_validation_signal','last_refit_label','first_execution','last_execution']:
                if row[key]!=reference[key]: raise AssertionError('Replica fold boundary changed')
            atomic_json(path,row); del model
            print(json.dumps({'model':cfg['id'],'month':fold['month'],'epochs':fit['best_epoch'],'seconds':row['seconds']}),flush=True)
    atomic_json(DEST/'training_matching.json',matches); print(json.dumps({'training_complete':True,'fits':len(matches)}),flush=True)


def prior_family():
    family,differences,returns,_=retained_family(); old=pd.read_csv(PRIOR/'joint_bootstrap.csv')
    for key,comp,origin in old[old.origin=='h10_control'][['id','comparator','origin']].drop_duplicates().itertuples(index=False,name=None):
        model,case=key.split('__',1)
        if comp=='cash': baseline=0.
        elif comp=='matched_mlp': baseline=returns(PRIOR,model.replace('cnn_','mlp_',1)+'__'+case)
        elif comp=='online_nonimage': baseline=returns(RANK_ROOT,'online_nonimage__'+case)
        else: raise AssertionError('Unknown retained comparator')
        family.append((key,comp,origin)); differences.append(returns(PRIOR,key)-baseline)
    if len(family)!=3224 or len(set(family))!=3224: raise AssertionError('Retained seed family incomplete')
    means=old[old.block==5].set_index(['id','comparator','origin'])['mean']
    error=max(abs(float(np.mean(x))-means.loc[key]) for key,x in zip(family,differences))
    if error>1e-14: raise AssertionError('Previous daily means changed')
    return family,differences,returns,error


def evaluate():
    spec=freeze(); torch.set_num_threads(4)
    files=sorted((DEST/'models').glob('*/*.pred.npy'))
    if len(files)!=36: raise AssertionError('Seed training incomplete')
    atomic_json(DEST/'forecast_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    p,ix,ci,di=context(SOURCE); p,release=amend_actions(p,ix,spec['training_parent']['actions'])
    p['sector_cap']=np.minimum(p['sector_cap'],.20); caps=historical_stock_caps(ix['codes'],ix['dates'])
    date_index={d:i for i,d in enumerate(ix['dates'])}; rows=[]; audits={}; reproductions=[]; baseline_checks=[]
    images=np.load(MODEL_ROOT/'images_raw.npy',mmap_mode='r'); x=torch.from_numpy(np.array(images[:,None],copy=True))
    for cfg in CONFIGS:
        path=DEST/'models'/cfg['id']/'202609.pt'; saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['config']!=cfg: raise AssertionError('Saved replica configuration changed')
        net=AlternativeNet(cfg['architecture']); net.load_state_dict(saved['state_dict'])
        expected=np.load(path.with_suffix('.pred.npy')); ids=np.flatnonzero(np.isfinite(expected)); actual=predict(net,x,ids); del net
        error=float(np.max(np.abs(actual-expected[ids])))
        if error>1e-6: raise AssertionError('Replica reinference differs')
        reproductions.append({'model':cfg['id'],'rows':len(ids),'maximum_error':error})
    del x
    folder=DEST/'portfolios'; folder.mkdir(exist_ok=True)
    for arch in ['cnn','mlp']:
        base=arch+'_pairwise_h10'; matrices={}
        for seed in [17,*SEEDS]:
            source=TRAIN_PRIOR/'models'/base if seed==17 else DEST/'models'/(base+f'_seed{seed}')
            matrices[f'seed{seed}']=score_matrix(merge_months(source,di,ix['dates']),ci,di,p['close'].shape)
        matrices['ensemble']=rank_consensus(matrices,{key:arch for key in matrices},p['eligible'],'mean')
        np.save(DEST/(arch+'_ensemble.npy'),matrices['ensemble'])
        for member,matrix in matrices.items():
            held={r:snapshot_scores(matrix,p['eligible'],ix['dates'],'20260101','20260923',r) for r in [1,5]}
            model=arch+'_rank_h10_'+member
            for case in CASES:
                alpha,origins=held[case['refresh']]; key=model+'__'+case['id']; path=folder/(key+'.json')
                if path.exists(): result=json.loads(path.read_text())
                else:
                    result=replay(p,ix,alpha,'20260101','20260923',top_n=case['top_n'],weight=.05,max_orders=case['max_orders'],
                                  rebalance=5,rebalance_band=.0005,return_trades=True,planning_price='open',
                                  action_release_dates=release,stock_cap_schedule=caps)
                    for trade in result['trades']:
                        d=date_index[trade['signal_date']]; origin=origins[d]
                        if not 0<=origin<=d: raise AssertionError('Future ensemble score')
                        trade['portfolio_score_date']=ix['dates'][origin]
                    result['metrics'].update(assess_weeks(result)); atomic_json(path,result)
                if member=='seed17':
                    original=json.loads((RANK_ROOT/'portfolios'/(base+'__'+case['id']+'.json')).read_text())
                    if result!=original: raise AssertionError('Seed17 baseline changed')
                    baseline_checks.append({'id':key,'all_fields_equal':True})
                audit=audit_fills(p,ix,result); audit.update(additional_checks(p,ix,result,None,max_orders=case['max_orders']))
                audit['announced_action_errors']=release_audit(p,ix,result,release)
                audit['dated_hynix_audit']=audit_pre_july_hynix(p,ix,result); audit['snapshot_origin_errors']=[]
                for trade in result['trades']:
                    d=date_index[trade['signal_date']]; origin=origins[d]
                    if not 0<=origin<=d or trade['portfolio_score_date']!=ix['dates'][origin]: audit['snapshot_origin_errors'].append(trade['date'])
                if (audit['post_buy_limit_violations'] or audit['additional_errors'] or audit['announced_action_errors']
                        or audit['dated_hynix_audit']['post_buy_limit_errors'] or audit['snapshot_origin_errors']
                        or audit['maximum_nav_reconstruction_error_krw']>.01):
                    atomic_json(DEST/'failed_audit.json',{'id':key,'audit':audit}); raise AssertionError('Seed replay audit failed')
                audits[key]=audit; blocks=calendar_blocks(result['daily'])
                rows.append(dict(id=key,model=model,architecture=arch,member=member,case=case['id'],**result['metrics'],
                                 **blocks,positive_blocks=sum(v>0 for v in blocks.values())))
            print(json.dumps({'model':model,'portfolios':len(rows)}),flush=True)
    if len(baseline_checks)!=16: raise AssertionError('Missing original seed accounts')
    atomic_json(DEST/'baseline_regression.json',baseline_checks); atomic_json(DEST/'checkpoint_reproduction.json',reproductions)
    atomic_json(DEST/'independent_audit.json',audits); frame=pd.DataFrame(rows); frame.to_csv(DEST/'portfolio_summary.csv',index=False)
    family,differences,returns,error=prior_family()
    atomic_json(DEST/'prior_family_reconstruction.json',{'hypotheses':len(family),'maximum_mean_error':error})
    new=frame[(frame.architecture=='cnn')&(frame.member!='seed17')]
    for row in new.to_dict('records'):
        for comp,baseline in [('cash',0.),('matched_mlp',returns(DEST,row['id'].replace('cnn_','mlp_',1))),
                              ('online_nonimage',returns(RANK_ROOT,'online_nonimage__'+row['case']))]:
            family.append((row['id'],comp,'rank_seeds')); differences.append(returns(DEST,row['id'])-baseline)
    if len(family)!=3296: raise AssertionError('Seed family changed')
    statistics=[]
    for block in [5,10]:
        result=family_bootstrap(np.column_stack(differences),block=block,draws=4000,seed=57)
        for i,(key,comp,origin) in enumerate(family):
            statistics.append(dict(id=key,comparator=comp,origin=origin,block=block,**{k:float(result[k][i]) for k in
                                   ['mean','standard_error','adjusted_p','marginal_p','simultaneous_lower95']}))
    stats=pd.DataFrame(statistics); stats.to_csv(DEST/'joint_bootstrap.csv',index=False); candidates=[]
    for row in new[new.member=='ensemble'].to_dict('records'):
        infer=stats[(stats.id==row['id'])&(stats.origin=='rank_seeds')]
        if len(infer)!=6: raise AssertionError('Incomplete ensemble comparisons')
        members=frame[(frame.architecture=='cnn')&(frame.case==row['case'])]
        if len(members)!=4: raise AssertionError('Missing seed stability member')
        stable=((members['return']>0)&(members.mdd>-.20)&(members.positive_blocks>=2)&(~members.four_week_turnover_stop)).all()
        if stable and (infer.adjusted_p<.025).all() and (infer.simultaneous_lower95>0).all(): candidates.append(row['id'])
    atomic_json(DEST/'evaluation_summary.json',{'status':'development_only','portfolios':len(rows),'new_portfolios':48,
                                               'seed17_regression_accounts':16,'joint_hypotheses':len(family),
                                               'robust_candidate_gate_passed':candidates,'independent_confirmation':False})
    print(json.dumps({'complete':True,'robust_candidates':candidates}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['freeze','train','evaluate','all']); args=ap.parse_args()
    if args.action=='freeze': freeze()
    if args.action in ['train','all']: train()
    if args.action in ['evaluate','all']: evaluate()

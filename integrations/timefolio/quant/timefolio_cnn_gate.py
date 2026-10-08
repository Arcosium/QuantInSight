"""Chronological cash-gate research from audited Stage-1 OOF predictions.

The gate requests exposure only. Holdings, fills, caps, and mandatory turnover
belong to the subsequent account replay and can make a cash policy ineligible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quant.timefolio_cnn_core import CashGate
from quant.timefolio_cnn_dataset import sha

FEATURES = ['universe_median_r5','universe_median_r19','breadth_above_ma20',
    'universe_median_daily_vol19','top10_mean_r5','top10_mean_r19',
    'top10_mean_daily_vol19','top10_score_gap_in_cross_section_sd']


def basket_features(closes, scores, keys, *, top_k=10):
    """Exactly 20 already observed closes; stock selection never sees labels."""
    c,s,k=map(np.asarray,[closes,scores,keys])
    if (c.shape!=(20,len(s)) or s.ndim!=1 or k.shape!=s.shape or len(s)<top_k
            or not np.isfinite(c).all() or (c<=0).any() or not np.isfinite(s).all()):
        raise ValueError('Complete observed windows and forecasts required')
    top=np.lexsort((k,-s))[:top_k]
    r5,r19=c[-1]/c[-6]-1,c[-1]/c[0]-1
    vol=np.diff(np.log(c),axis=0).std(axis=0,ddof=1)
    spread=float(s.std())
    confidence=(s[top].mean()-s.mean())/spread if spread>1e-12 else 0.
    x=np.array([np.median(r5),np.median(r19),np.mean(c[-1]>c.mean(axis=0)),
        np.median(vol),r5[top].mean(),r19[top].mean(),vol[top].mean(),confidence])
    if not np.isfinite(x).all():raise ValueError('Nonfinite gate feature')
    return top,x


def net_roundtrip(gross, *, horizon=5, cash_annual=0.):
    """Conservative all-new basket: buy10bp, sell30bp, slippage5bp each way."""
    if horizon<1 or not np.isfinite(cash_annual) or cash_annual<0:
        raise ValueError('Invalid holding period/cash opportunity rate')
    return (1+np.asarray(gross))*(1-.0005)*(1-.003)/((1+.0005)*(1+.001))-1-((1+cash_annual)**(horizon/252)-1)


def walk_gate(x, net_y, signal, label_end, train_end, selection_end, folds, *, days, minimum_rows=60):
    x,y,s,end,train,selection=map(np.asarray,[x,net_y,signal,label_end,train_end,selection_end])
    if (x.shape!=(len(s),len(FEATURES)) or any(v.shape!=s.shape for v in [y,end,train,selection])
            or len(np.unique(s))!=len(s) or not np.isfinite(x).all()
            or (train>=s).any() or (selection>=s).any() or (end<=s).any()):
        raise ValueError('Unique chronological OOF basket records required')
    expected=np.full(days,np.nan);cold=np.zeros(days,bool); fits=[]
    exposures={name:np.full(days,np.nan) for name in ['always','cash','trend_floor20','ridge_cash','ridge_floor20']}
    for fold in folds:
        cutoff=fold['test_start'];forecast=(s>=cutoff)&(s<fold['test_end'])
        if not forecast.any():continue
        matured=(s<cutoff)&(end<cutoff)&np.isfinite(y)
        if matured.sum()>=minimum_rows:
            model=CashGate.fit(x[matured],y[matured],signal_dates=s[matured],
                feature_available_dates=s[matured],label_available_dates=end[matured],
                stage1_train_label_end=train[matured],stage1_selection_label_end=selection[matured],
                fit_cutoff=cutoff,alpha=10.,minimum_rows=minimum_rows)
            estimate=model.predict(x[forecast],signal_dates=s[forecast],feature_available_dates=s[forecast])
            fits.append(dict(fold=fold['id'],fit_cutoff=int(cutoff),matured_dates=int(matured.sum()),
                last_label_end=int(end[matured].max()),mean=model.mean.tolist(),scale=model.scale.tolist(),
                coefficient=model.coefficient.tolist(),intercept=model.intercept,cold_start=False))
        else:
            estimate=np.full(int(forecast.sum()),np.nan);cold[s[forecast]]=True
            fits.append(dict(fold=fold['id'],fit_cutoff=int(cutoff),matured_dates=int(matured.sum()),
                cold_start=True,fallback='always invested until60 matured OOF baskets'))
        ix=s[forecast];expected[ix]=estimate
        exposures['always'][ix]=.8;exposures['cash'][ix]=0.
        exposures['trend_floor20'][ix]=np.where(x[forecast,1]>0,.8,.2)
        # Cold start is declared in advance and shared by learned gate variants.
        exposures['ridge_cash'][ix]=np.where(np.isnan(estimate)|(estimate>0),.8,0.)
        exposures['ridge_floor20'][ix]=np.maximum(exposures['ridge_cash'][ix],.2)
    return expected,cold,exposures,fits


def build(dataset, results, output, objective, *, allow_partial=False, seed=17):
    dataset,results,output=map(Path,[dataset,results,output])
    if type(seed) is not int or seed not in (17,29,43):
        raise ValueError('Registered CNN seed required')
    if output.exists():raise ValueError('Refusing to overwrite frozen gate results')
    manifest=json.loads((dataset/'manifest.json').read_text());arrays={}
    if (dataset/'RETIRED.json').exists() or manifest.get('exploratory_ready') is not True:
        raise ValueError('Consistent-price exploratory dataset required')
    for name in ['daily_ohlcv','returns','signal_index','label_end_index','security_key','eligible','dates','codes']:
        spec=manifest['arrays'][name];p=(dataset/spec['path']).resolve()
        if not p.is_relative_to(dataset.resolve()) or sha(p)!=spec['sha256']:
            raise ValueError('Dataset array changed')
        arrays[name]=np.load(p,mmap_mode='r',allow_pickle=False)
    index=arrays['signal_index'];keys=arrays['security_key'];labels=arrays['returns']
    dates=arrays['dates'];codes=arrays['codes'];close=arrays['daily_ohlcv'][:,:,3]
    matrix=np.full((len(dates),len(codes)),np.nan,dtype=np.float32)
    records=[];receipts=[];completed=[]
    for fold in manifest['folds']:
        directory=results/f"{fold['id']}_{objective}_{seed}";receipt=directory/'receipt.json'
        if not receipt.exists():
            if allow_partial:break  # a contiguous prefix only; never skip an unavailable month
            raise ValueError('Missing registered monthly model: '+fold['id'])
        proof=json.loads(receipt.read_text())
        if (proof['dataset_manifest_sha256']!=sha(dataset/'manifest.json') or proof['fold']!=fold
                or proof['config']['objective']!=objective or proof['config']['seed']!=seed):
            raise ValueError('Model identity differs from the registered OOF job')
        for file in ['sample_ids.npy','scores.npy']:
            if sha(directory/file)!=proof['artifacts'][file]:raise ValueError('Forecast fingerprint changed')
        ids=np.load(directory/'sample_ids.npy',allow_pickle=False);scores=np.load(directory/'scores.npy',allow_pickle=False)
        expected=np.flatnonzero(arrays['eligible']&(index>=fold['test_start'])&(index<fold['test_end']))
        if not np.array_equal(ids,expected) or scores.shape!=ids.shape or not np.isfinite(scores).all():
            raise ValueError('Prediction membership was changed or a score is missing')
        last_train,last_selection=proof['last_training_label_index'],proof['last_selection_label_index']
        if proof.get('last_selection_account_mark_index') is not None:
            last_selection=max(last_selection,proof['last_selection_account_mark_index'])
        if max(last_train,last_selection)>=int(index[ids].min()):raise ValueError('Stage1 lookahead')
        matrix[index[ids],keys[ids]]=scores
        for day in np.unique(index[ids]):
            on=index[ids]==day;ix=ids[on];stock=keys[ix]
            top,x=basket_features(close[day-19:day+1,stock],scores[on],stock)
            realized=labels[ix[top]]
            gross=float(realized.mean()) if np.isfinite(realized).all() else None
            records.append(dict(signal=int(day),date=str(dates[day]),label_end=int(day+manifest['horizon']+1),
                features=x.tolist(),gross_holding_return=gross,
                net_excess=float(net_roundtrip(gross,horizon=manifest['horizon'])) if gross is not None else None,
                stage1_train_label_end=last_train,stage1_selection_label_end=last_selection,
                selected_codes=[str(codes[i]) for i in stock[top]],fold=fold['id']))
        receipts.append(dict(fold=fold['id'],receipt_sha256=sha(receipt)));completed.append(fold)
    if not records:raise ValueError('No complete monthly forecasts')
    x=np.array([r['features'] for r in records]);s=np.array([r['signal'] for r in records])
    y=np.array([r['net_excess'] if r['net_excess'] is not None else np.nan for r in records])
    end=np.array([r['label_end'] for r in records]);train=np.array([r['stage1_train_label_end'] for r in records]);selection=np.array([r['stage1_selection_label_end'] for r in records])
    expected,cold,exposures,fits=walk_gate(x,y,s,end,train,selection,completed,days=len(dates))
    output.mkdir(parents=True)
    np.save(output/'scores.npy',matrix,allow_pickle=False)
    np.savez(output/'gate.npz',signal_index=s,features=x,net_excess=y,label_end=end,
        expected_net=expected,cold_start=cold,**{'gross_'+k:v for k,v in exposures.items()})
    (output/'baskets.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (output/'fits.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    proof=dict(objective=objective,seed=seed,features=FEATURES,horizon=manifest['horizon'],
        dataset_manifest_sha256=sha(dataset/'manifest.json'),model_receipts=receipts,
        completed_folds=len(completed),registered_folds=len(manifest['folds']),
        all_folds_complete=len(completed)==len(manifest['folds']),basket_dates=len(records),
        gate_alpha=10.,minimum_matured_dates=60,cold_start_policy='always invested',
        cost_policy='Full roundtrip buy10bp, sell30bp, slippage5bp per side; cash interest/opportunity baseline0.',
        gate_predictions_are_account_exposure_requests_only=True,account_validation_complete=False,
        source_sha256=sha(__file__),artifacts={p.name:sha(p) for p in output.iterdir()})
    (output/'receipt.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(objective=objective,folds=len(completed),dates=len(records),
        learned_gate_folds=sum(not f['cold_start'] for f in fits))),flush=True)
    return proof


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--results',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--objective',choices=['mse','bce','listnet','topk_ce'],required=True)
    ap.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    build(args.dataset,args.results,args.output,args.objective,allow_partial=args.allow_partial,seed=args.seed)

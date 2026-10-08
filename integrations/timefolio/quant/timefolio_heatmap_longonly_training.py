"""Monthly heatmap forecasts trained on absolute, after-cost five-session outcomes.

New development hypotheses after prior allocation results. The target explicitly
includes both commissions and slippage; known action corrections are shared by
all neural and nonimage controls. No new independent holdout is claimed.
"""
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
from lightgbm import LGBMClassifier,LGBMRegressor,early_stopping,log_evaluation

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import DEST as ACTION_ROOT,amend_actions
from quant.timefolio_heatmap_features import labels
from quant.timefolio_heatmap_replay import BUY_FEE,SELL_FEE
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import DEST as MODEL_ROOT,SOURCE,fold_masks,monthly_folds,train_nn,predict

DEST=SOURCE.with_name('20260928_longonly_training_v1')
CONFIGS=[dict(id=f'{architecture}_net_{mode}_h5',architecture=architecture,
              mode=mode,target='binary' if mode=='binary' else 'absolute',horizon=5,
              encoding='raw',train_window=0,epochs=6,lr=.0007,seed=17)
         for architecture in ['cnn','gbm'] for mode in ['binary','return']]


def net_targets(forward,mode,slip=.0005):
    """A unit-cash round trip, including fees on the slipped purchase/sale."""
    forward=np.asarray(forward,float)
    net=(1+forward)*(1-slip)*(1-SELL_FEE)/((1+slip)*(1+BUY_FEE))-1
    if mode=='binary':out=(net>0).astype(float)
    elif mode=='return':out=np.clip(net/.10,-3,3)
    else:raise ValueError('Unknown long-only target')
    out[~np.isfinite(net)]=np.nan
    return out.astype(np.float32),net


def freeze():
    parent=json.loads((ACTION_ROOT/'protocol.json').read_text())
    files=[Path(__file__),ACTION_ROOT/'protocol.json',MODEL_ROOT/'images_raw.npy',SOURCE/'panel.npz',SOURCE/'samples.npz']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in ['walkforward','action_amendment','study','features','data','replay']]
    spec={'configs':CONFIGS,'corporate_actions':parent['actions'],'source':str(SOURCE),
          'base_celltrion_override':'same corrected_panel as the v5 ledger',
          'labels':'recompute cumulative factors from the known corrected share events; five-session forward mark from next execution-window entry; net factor=(1-slip)*(1-sellfee)/((1+slip)*(1+buyfee))',
          'costs':{'buy_fee':BUY_FEE,'sell_fee':SELL_FEE,'slippage_each_side':.0005},
          'binary':'net return > 0; no daily/sector median subtraction',
          'regression':'net return divided by .10, clipped to [-3,3]',
          'training':'same nine monthly folds and purged past inner validation; CNN BCE/AP or Huber/daily-IC; GBM classifier/AP or regressor/L2',
          'inputs':'frozen raw heatmap; 96 summary features for GBM; input images themselves are unchanged',
          'evaluation_plan':'top4/top12, 3/10 daily orders, stock5%, sector20%; always rank versus positive forecast gate; corresponding label-matched GBM; append to prior 1032-hypothesis family',
          'selection_context':'devised after previous full allocation outcomes; adaptive development, independent confirmation absent',
          'limitations':'remaining unverified actions/dividends, current GICS and survivor universe, approximate fills and overlap of five-session labels',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Long-only training protocol changed')
    if not path.exists():
        atomic_json(path,spec);folder=DEST/'frozen_source';folder.mkdir(exist_ok=True)
        for i,p in enumerate(files):
            if p.suffix in ['.py','.json']:shutil.copy2(p,folder/(f'{i:02d}_'+p.name))
    return spec


def run():
    spec=freeze();torch.set_num_threads(4)
    p,ix,ci,di=context(SOURCE);p,_=amend_actions(p,ix,spec['corporate_actions'])
    p['factor']=np.cumprod(p['split'],axis=1);_,forward=labels(p,ci,di,5)
    dates=np.asarray(ix['dates']);allowed=np.zeros(len(ci),bool);valid=di+1<len(dates)
    allowed[valid]=p['trade_allowed'][ci[valid],di[valid]+1].astype(bool)
    raw=np.load(MODEL_ROOT/'images_raw.npy',mmap_mode='r')
    for cfg in CONFIGS:
        folder=DEST/'models'/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
        y,net=net_targets(forward,cfg['mode']);label_path=DEST/f"labels_{cfg['mode']}.npy"
        if label_path.exists():
            if not np.array_equal(np.load(label_path),y,equal_nan=True):raise RuntimeError('Labels changed')
        else:np.save(label_path,y)
        if cfg['architecture']=='cnn':x=torch.from_numpy(np.array(raw[:,None],copy=True))
        else:x=np.concatenate([raw[:,:,-1],raw.mean(2),raw.std(2)],1).astype(np.float32)/255
        for fold in monthly_folds(ix['dates']):
            path=folder/(fold['month']+'.json')
            if path.exists():
                if not path.with_suffix('.pred.npy').exists():raise RuntimeError('Incomplete long-only checkpoint')
                continue
            started=time.monotonic();start,end=fold['start'],fold['end']
            train,val,refit,pred,_=fold_masks(di,y,allowed,5,start,end)
            if train.sum()<1000 or val.sum()<200:raise RuntimeError('Insufficient causal observations')
            if cfg['architecture']=='cnn':
                model,fit=train_nn(x,y,train,val,cfg,di);del model
                model,_=train_nn(x,y,refit,np.zeros(len(y),bool),cfg,di,epochs=fit['best_epoch'])
                values=predict(model,x,np.flatnonzero(pred),cfg['mode']=='binary')
                torch.save({'config':cfg,'state_dict':model.state_dict()},path.with_suffix('.pt'))
            else:
                binary=cfg['mode']=='binary'
                def estimator(n):
                    cls=LGBMClassifier if binary else LGBMRegressor
                    return cls(n_estimators=n,num_leaves=15,learning_rate=.03,min_child_samples=100,
                               reg_lambda=10.,subsample=.8,subsample_freq=1,colsample_bytree=.8,
                               random_state=17,n_jobs=4,verbosity=-1,metric='average_precision' if binary else 'l2')
                model=estimator(250);model.fit(x[train],y[train],eval_set=[(x[val],y[val])],callbacks=[early_stopping(25,verbose=False),log_evaluation(0)])
                fit={'best_epoch':max(1,int(model.best_iteration_)),'training_rows':int(train.sum()),'validation_rows':int(val.sum())}
                model=estimator(fit['best_epoch']);model.fit(x[refit],y[refit])
                values=model.predict_proba(x[pred])[:,1] if binary else model.predict(x[pred])
                model.booster_.save_model(str(path.with_suffix('.txt')))
            if not np.isfinite(values).all():raise AssertionError('Non-finite predictions')
            if cfg['mode']=='binary' and np.any((values<0)|(values>1)):raise AssertionError('Invalid binary scores')
            out=np.full(len(ci),np.nan,np.float32);out[pred]=values;np.save(path.with_suffix('.pred.npy'),out)
            row={'config':cfg,'fold':fold,'fit':fit,'refit_rows':int(refit.sum()),
                 'last_refit_label':str(max(dates[di[refit]+5])),'first_execution':ix['dates'][start],
                 'last_execution':ix['dates'][end-1],'seconds':round(time.monotonic()-started,2),
                 'refit_net_positive_fraction':float(np.mean(net[refit]>0))}
            if row['last_refit_label']>=row['first_execution']:raise AssertionError('Future labels')
            atomic_json(path,row);del model
            print(json.dumps({'model':cfg['id'],'month':fold['month'],'epochs':fit['best_epoch'],'seconds':row['seconds']}),flush=True)
        del x
    print(json.dumps({'training_complete':True,'fits':36}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:run()

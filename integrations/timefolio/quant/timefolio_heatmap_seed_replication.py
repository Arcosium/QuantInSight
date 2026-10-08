"""Predeclared seed replication of the early, exploratory CNN H5 candidate.

The candidate was noticed in the planning probe; these are robustness results on
reused history. No seed is selected for its realised portfolio performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import (
    DEST as MODEL_ROOT, SOURCE, monthly_folds, fold_masks, continuous_targets,
    train_nn, predict, daily_ic,
)

DEST=SOURCE.with_name('20260928_seed_replication_v1')
SEEDS=[29,43]
MODEL='cnn_sector_h5'


def freeze():
    parent=json.loads((MODEL_ROOT/'protocol.json').read_text())
    cfg=next(x for x in parent['configs'] if x['id']==MODEL)
    files=[Path(__file__),MODEL_ROOT/'protocol.json',MODEL_ROOT/'images_raw.npy',SOURCE/'panel.npz',SOURCE/'samples.npz']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in ['walkforward','study','features','data']]
    spec={'model':MODEL,'base_config':cfg,'new_seeds':SEEDS,'existing_seed':17,
          'origins':'same nine monthly causal folds as v3; past inner validation selects epoch separately for each seed',
          'selection_context':'chosen after observing the initial planning probe; before full v3/v4/v5 portfolio outcomes',
          'evaluation':'all three individual seeds plus fixed equal average of within-date ranks; no best-seed selection',
          'inference':'append new seed29/43/ensemble hypotheses to the v5 family; same four policies and three budgets; no reset of comparison count',
          'status':'development robustness only; independent confirmation still absent',
          'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
          'original_forecasts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((MODEL_ROOT/'models'/MODEL).glob('*.pred.npy'))}}
    if len(spec['original_forecasts'])!=9:raise RuntimeError('Original candidate is incomplete')
    DEST.mkdir(exist_ok=True);path=DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise RuntimeError('Seed replication protocol changed')
    if not path.exists():
        atomic_json(path,spec);folder=DEST/'frozen_source';folder.mkdir(exist_ok=True)
        for p in files:
            if p.suffix=='.py':shutil.copy2(p,folder/p.name)
    return spec


def run():
    spec=freeze();torch.set_num_threads(1)
    atomic_json(DEST/'effective_runtime.json',{'cpu_affinity':sorted(os.sched_getaffinity(0)),
                                            'torch_threads':torch.get_num_threads(),
                                            'numerical_thread_env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}})
    if len(os.sched_getaffinity(0))!=1:raise RuntimeError('Replication must run on one CPU')
    p,ix,ci,di=context(SOURCE);dates=np.asarray(ix['dates'])
    allowed=np.zeros(len(ci),bool);valid=di+1<len(dates)
    allowed[valid]=p['trade_allowed'][ci[valid],di[valid]+1].astype(bool)
    raw=np.load(MODEL_ROOT/'images_raw.npy',mmap_mode='r');x=torch.from_numpy(np.array(raw[:,None],copy=True))
    base=spec['base_config'];y=continuous_targets(p,ci,di,base['horizon'],base['target'])
    common=continuous_targets(p,ci,di,3,'sector')
    for seed in SEEDS:
        cfg=dict(base,seed=seed);folder=DEST/'models'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path=folder/(fold['month']+'.json')
            if path.exists():
                if not path.with_suffix('.pred.npy').exists() or not path.with_suffix('.pt').exists():raise RuntimeError('Incomplete replica checkpoint')
                continue
            started=time.monotonic();start,end=fold['start'],fold['end']
            tr,va,refit,pr,inner=fold_masks(di,y,allowed,cfg['horizon'],start,end,cfg['train_window'])
            if tr.sum()<1000 or va.sum()<200:raise RuntimeError('Insufficient causal data')
            model,fit=train_nn(x,y,tr,va,cfg,di)
            ins=np.full(len(ci),np.nan,np.float32);ins[inner]=predict(model,x,np.flatnonzero(inner))
            mask=inner&allowed&(di+3<start)&np.isfinite(common)
            ic=daily_ic(common,ins,mask,di);del model
            model,_=train_nn(x,y,refit,np.zeros(len(y),bool),cfg,di,epochs=fit['best_epoch'])
            values=np.full(len(ci),np.nan,np.float32);values[pr]=predict(model,x,np.flatnonzero(pr))
            torch.save({'config':cfg,'state_dict':model.state_dict()},path.with_suffix('.pt'))
            np.save(path.with_suffix('.pred.npy'),values)
            row={'config':cfg,'fold':fold,'fit':fit,'common_inner_ic':ic,'refit_rows':int(refit.sum()),
                 'last_refit_label':str(max(dates[di[refit]+cfg['horizon']])),
                 'first_execution':ix['dates'][start],'last_execution':ix['dates'][end-1],
                 'seconds':round(time.monotonic()-started,2)}
            if row['last_refit_label']>=row['first_execution']:raise AssertionError('Future refit label')
            atomic_json(path,row);del model
            print(json.dumps({'seed':seed,'month':fold['month'],'epochs':fit['best_epoch'],'inner_ic':ic,'seconds':row['seconds']}),flush=True)
    print(json.dumps({'seed_training_complete':True,'fits':18}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','run']);a=ap.parse_args()
    if a.action=='freeze':freeze()
    else:run()

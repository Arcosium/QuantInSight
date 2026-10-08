"""Date-balanced pointwise and pairwise learning on frozen KRX heatmaps.

RankNet comparisons are confined to a signal date. All labels, model selection,
and refits retain monthly purges; this is development on previously seen dates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_action_amendment import DEST as ACTION_ROOT, amend_actions
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import (
    SOURCE, DEST as MODEL_ROOT, AlternativeNet, continuous_targets, monthly_folds,
    fold_masks, daily_ic, predict,
)

DEST = SOURCE.with_name('20260929_rank_training_v1')
CONFIGS = [dict(id=f'{arch}_{objective}_h{h}', architecture=arch, objective=objective,
                horizon=h, target='sector', seed=17, lr=.0007, epochs=6)
           for arch in ['cnn', 'mlp'] for objective, h in [('huber', 5), ('pairwise', 5), ('pairwise', 10)]]
REFERENCE = 'https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf'


def pairwise_loss(scores, target):
    """Mean logistic loss over all strict winner/loser pairs of ONE date."""
    if scores.ndim != 1 or target.shape != scores.shape:
        raise ValueError('One equally sized score/target vector required')
    if not torch.isfinite(scores).all() or not torch.isfinite(target).all():
        raise ValueError('Finite pairwise scores and targets required')
    winning = target[:, None] > target[None, :]
    if not winning.any(): return scores.sum()*0
    margin = scores[:, None]-scores[None, :]
    return nn.functional.softplus(-margin[winning]).mean()


def date_groups(di, mask):
    di, mask = np.asarray(di), np.asarray(mask, dtype=bool)
    if di.ndim != 1 or mask.shape != di.shape: raise ValueError('Matching date and mask vectors required')
    return [np.flatnonzero(mask & (di == d)) for d in np.unique(di[mask])]


def train_daywise(x, y, train, val, cfg, di, *, epochs=None):
    seed = cfg['seed']; torch.manual_seed(seed)
    model = AlternativeNet(cfg['architecture'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    groups = date_groups(di, train); va = np.flatnonzero(val)
    if not groups or max(map(len, groups)) > 512: raise ValueError('Missing or unexpectedly large date group')
    if not np.isfinite(y[train | val]).all(): raise ValueError('Non-finite training label')
    best = -np.inf; state = None; best_epoch = 0; history = []
    if cfg['objective'] not in ['pairwise', 'huber']: raise ValueError('Unknown ranking objective')
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses = []
        for group in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ids = groups[group]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1))
            loss = (pairwise_loss(out, target[ids]) if cfg['objective'] == 'pairwise'
                    else nn.functional.huber_loss(out, target[ids], delta=1.))
            if not torch.isfinite(loss): raise AssertionError('Non-finite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            losses.append(float(loss.detach()))
        metric = 0.
        if len(va):
            values = np.full(len(y), np.nan); values[va] = predict(model, x, va)
            metric = daily_ic(y, values, val, di)
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses)), 'inner_ic': metric})
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch+1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {'best_epoch': best_epoch, 'history': history, 'training_rows': int(train.sum()),
                   'validation_rows': int(val.sum()), 'training_dates': len(groups)}


def freeze():
    parent = json.loads((ACTION_ROOT/'protocol.json').read_text())
    files = [Path(__file__), ACTION_ROOT/'protocol.json', MODEL_ROOT/'images_raw.npy',
             SOURCE/'panel.npz', SOURCE/'panel_index.json', SOURCE/'samples.npz']
    files += [Path(__file__).with_name('timefolio_heatmap_'+n+'.py') for n in
              ['walkforward', 'action_amendment', 'execution_amendment', 'study', 'features', 'data', 'replay']]
    spec = {'configs': CONFIGS, 'actions': parent['actions'], 'reference': REFERENCE,
            'evaluation_start': '20260101', 'evaluation_end': '20260923',
            'inputs': 'same raw32x65 per-stock images; CNN unchanged; MLP uses same96 time summaries as previous alternative model',
            'labels': 'corrected cumulative share factors; existing sector-residual horizon5/10 labels divided0.10 and clipped[-3,3]; sector median when>=4 observations, else daily market median',
            'pairwise': 'all strict winner/loser pairs within a single signal date, mean softplus(-(winner_score-loser_score)); ties omitted; never compare samples across dates',
            'training': 'one whole signal-date batch per step for BOTH objectives; equal date weight, AdamW lr0.0007 decay0.001, gradient clip5; max6 epochs selected on purged past20-session inner dailyIC; restart and refit observed labels',
            'pointwise_control': 'H5 Huber shares exactly the pairwise H5 data, network, date batches, seed, optimiser and selection metric; only loss changes',
            'evaluation_plan': 'all6 forecasts; target5%, sector20%, gross80%; top4/12 x3/10dailyfills x daily/5session-score snapshot, rebalance5 and NAV5bp band; dated caps and completed-holiday-week turnover; cash, matchedMLP and online_nonimage comparisons for every CNN',
            'selection_context': 'new ranking hypothesis after original3128 full-period comparisons; append all72 new comparisons rather than retain only winning forecasts; family alpha0.025 reserves remaining0.025 for separate shorter news study',
            'limitations': 'historical development; ranking does not ensure positive absolute after-cost return; prior unverified actions/dividends, currentGICS/survivor universe, and approximate order-book execution remain; no fresh holdout viewed',
            'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    DEST.mkdir(exist_ok=True); path = DEST/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Rank training specification changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST/'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix in ['.py', '.json']: shutil.copy2(p, folder/(f'{i:02d}_'+p.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    p, ix, ci, di = context(SOURCE); p, _ = amend_actions(p, ix, spec['actions'])
    p['factor'] = np.cumprod(p['split'], axis=1)
    dates = np.asarray(ix['dates']); valid = di+1 < len(dates)
    allowed = np.zeros(len(ci), bool); allowed[valid] = p['trade_allowed'][ci[valid], di[valid]+1].astype(bool)
    raw = np.load(MODEL_ROOT/'images_raw.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(raw[:, None], copy=True))
    for cfg in CONFIGS:
        folder = DEST/'models'/cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        y = continuous_targets(p, ci, di, cfg['horizon'], 'sector')
        label_path = DEST/f"labels_h{cfg['horizon']}.npy"
        if label_path.exists():
            if not np.array_equal(np.load(label_path), y, equal_nan=True): raise AssertionError('Labels changed')
        else: np.save(label_path, y)
        for fold in monthly_folds(ix['dates']):
            path = folder/(fold['month']+'.json')
            if path.exists():
                if not path.with_suffix('.pred.npy').exists() or not path.with_suffix('.pt').exists():
                    raise RuntimeError('Incomplete rank checkpoint')
                continue
            started = time.monotonic(); start, end = fold['start'], fold['end']
            train, val, refit, pred, _ = fold_masks(di, y, allowed, cfg['horizon'], start, end)
            if train.sum() < 1000 or val.sum() < 200: raise RuntimeError('Insufficient purged observations')
            model, fit = train_daywise(x, y, train, val, cfg, di); del model
            model, _ = train_daywise(x, y, refit, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            values = predict(model, x, np.flatnonzero(pred))
            if not np.isfinite(values).all(): raise AssertionError('Non-finite ranking predictions')
            out = np.full(len(ci), np.nan, np.float32); out[pred] = values
            torch.save({'config': cfg, 'state_dict': model.state_dict()}, path.with_suffix('.pt'))
            np.save(path.with_suffix('.pred.npy'), out)
            row = {'config': cfg, 'fold': fold, 'fit': fit, 'refit_rows': int(refit.sum()),
                   'last_inner_train_label': str(max(dates[di[train]+cfg['horizon']])),
                   'first_inner_validation_signal': str(min(dates[di[val]])),
                   'last_refit_label': str(max(dates[di[refit]+cfg['horizon']])),
                   'first_execution': ix['dates'][start], 'last_execution': ix['dates'][end-1],
                   'seconds': round(time.monotonic()-started, 2)}
            if (row['last_refit_label'] >= row['first_execution']
                    or row['last_inner_train_label'] >= row['first_inner_validation_signal']):
                raise AssertionError('Labels crossed a partition')
            atomic_json(path, row); del model
            print(json.dumps({'model': cfg['id'], 'month': fold['month'],
                              'epochs': fit['best_epoch'], 'seconds': row['seconds']}), flush=True)
    print(json.dumps({'training_complete': True, 'fits': 54}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

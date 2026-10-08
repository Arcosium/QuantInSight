"""Two-channel CNN/MLP with matched blank-context controls and three seeds."""
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
from quant.timefolio_heatmap_peer_features import DEST as PEER_ROOT, STOCK_ROOT
from quant.timefolio_heatmap_rank_training import DEST as LABEL_ROOT, pairwise_loss, date_groups
from quant.timefolio_heatmap_corrected_evaluation import DEST as PRIOR
from quant.timefolio_heatmap_rank_evaluation import CASES
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import (
    SOURCE, continuous_targets, monthly_folds, fold_masks, daily_ic, predict,
)

DEST = SOURCE.with_name('20260929_peer_training_v1')
CONFIGS = [dict(id=f'{arch}_{mode}_h10_seed{seed}', architecture=arch, context_mode=mode,
                objective='pairwise', horizon=10, target='sector', seed=seed, lr=.0007, epochs=6)
           for seed in [17, 29, 43] for arch in ['cnn', 'mlp'] for mode in ['blank', 'peer']]
EVALUATION_PLAN = dict(
    cases=CASES, buffer_multipliers=[0, 1, 2], accounts=504, new_hypotheses=992, joint_hypotheses=4896,
    streams='12new forecasts,4 fixed rank ensembles,4old corrected raw CNN members,1fixed legacy online_nonimage; all21 streams x8cases x3buffers',
    comparisons='Every new CNN vs cash, matched same-context MLP, legacy online and corrected raw CNN of same seed/ensemble. Peer CNN also vs blank CNN. Positive buffers additionally vs same new CNN at buffer0. All same execution case.',
    gate='Only fixed ensembles can qualify. Each ensemble and its3seeds must have positive net return, MDD>-20%, at least2positive quarters and fewer than4turnover failures. Every comparator at both blocks must pass adjustedp<0.025 and simultaneous_lower95>0. Stress and new independent confirmation still required.',
    bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
    interpretation='Reused development. Retain3904 earlier hypotheses; blank CNN adds448 and peer CNN544. No seed or context choice based on future account outcomes. News75 remain separate.')


class ContextNet(nn.Module):
    def __init__(self, architecture, context_mode):
        super().__init__()
        if context_mode not in ['blank', 'peer']: raise ValueError('Unknown context mode')
        self.architecture = architecture; self.context_mode = context_mode
        if architecture == 'cnn':
            self.net = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.LeakyReLU(.1), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.LeakyReLU(.1),
                nn.AdaptiveAvgPool2d((2, 4)), nn.Flatten(), nn.Dropout(.1), nn.Linear(256, 1))
        elif architecture == 'mlp':
            self.net = nn.Sequential(nn.Linear(192, 32), nn.LeakyReLU(.1), nn.Dropout(.1),
                                     nn.Linear(32, 16), nn.LeakyReLU(.1), nn.Linear(16, 1))
        else: raise ValueError('Unknown peer architecture')

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:3] != (2, 32): raise ValueError('Two32-row channels required')
        if self.context_mode == 'blank': x = torch.cat([x[:, :1], torch.zeros_like(x[:, 1:])], 1)
        if self.architecture == 'mlp':
            x = torch.cat([x[..., -1].flatten(1), x.mean(-1).flatten(1), x.std(-1, correction=0).flatten(1)], 1)
        return self.net(x).squeeze(1)


def train_context(x, y, train, val, cfg, di, *, epochs=None):
    seed = cfg['seed']; torch.manual_seed(seed)
    model = ContextNet(cfg['architecture'], cfg['context_mode'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    groups = date_groups(di, train); va = np.flatnonzero(val)
    if not groups or max(map(len, groups)) > 512: raise ValueError('Invalid signal-date groups')
    if cfg['objective'] != 'pairwise' or not np.isfinite(y[train | val]).all(): raise ValueError('Invalid context objective/labels')
    best = -np.inf; state = None; best_epoch = 0; history = []
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses = []
        for group in np.random.default_rng(seed + epoch).permutation(len(groups)):
            ids = groups[group]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1)); loss = pairwise_loss(out, target[ids])
            if not torch.isfinite(loss): raise AssertionError('Non-finite peer loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            losses.append(float(loss.detach()))
        metric = 0.
        if len(va):
            values = np.full(len(y), np.nan); values[va] = predict(model, x, va)
            metric = daily_ic(y, values, val, di)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), inner_ic=metric))
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, history=history, training_rows=int(train.sum()),
                       validation_rows=int(val.sum()), training_dates=len(groups))


def freeze():
    peer = json.loads((PEER_ROOT / 'protocol.json').read_text())
    for manifest in [peer['hashes'], json.loads((PEER_ROOT / 'data_hashes.json').read_text())]:
        for filename, value in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value: raise AssertionError('Peer input changed')
    files = [Path(__file__), PEER_ROOT / 'protocol.json', PEER_ROOT / 'data_hashes.json',
             LABEL_ROOT / 'labels_h10.npy', PRIOR / 'protocol.json', PRIOR / 'joint_bootstrap.csv']
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['peer_features', 'corrected_features', 'rank_training', 'rank_evaluation', 'walkforward', 'study', 'features', 'data']]
    spec = dict(configs=CONFIGS, peer_protocol=peer, planned_fits=108, evaluation_plan=EVALUATION_PLAN,
                architecture='CNN unchanged8/16/32convolution and2x4pool/head except2input channels. MLP same32/16head with192summary inputs. Blank and peer within each architecture/seed have identical initial tensors and parameter count; blank zeros second normalized channel inside forward.',
                training='Same corrected H10 sector-residual labels; one date per step, pairwise loss, AdamW0.0007 decay0.001 clip5, max6epochs selected by purged past20session inner dailyIC. Restart/refit all labels known before monthly first execution. All3seeds kept.',
                limitations='Extra same-date context and larger first-layer capacity are separated by matched blank controls and prior corrected raw CNN. Static sector mapping/survivor universe, corporate-action/dividend and execution limitations remain. No reserved September28 outcomes.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Peer training protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:02d}_' + p.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    p, ix, ci, di = context(STOCK_ROOT); dates = np.asarray(ix['dates'])
    p['factor'] = np.cumprod(p['split'], axis=1); y = continuous_targets(p, ci, di, 10, 'sector')
    if not np.array_equal(y, np.load(LABEL_ROOT / 'labels_h10.npy'), equal_nan=True): raise AssertionError('Peer labels changed')
    allowed = np.zeros(len(ci), bool); valid_next = di + 1 < len(dates)
    allowed[valid_next] = p['trade_allowed'][ci[valid_next], di[valid_next] + 1].astype(bool)
    raw = np.load(STOCK_ROOT / 'images_raw.npy', mmap_mode='r'); peer = np.load(PEER_ROOT / 'images_peer.npy', mmap_mode='r')
    x = torch.from_numpy(np.stack([raw, peer], axis=1)); completed = []
    for cfg in CONFIGS:
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if path.exists(): raise RuntimeError('Do not overwrite a completed peer fit')
            started = time.monotonic(); start, end = fold['start'], fold['end']
            train, val, refit, pred, _ = fold_masks(di, y, allowed, 10, start, end)
            if train.sum() < 1000 or val.sum() < 200: raise RuntimeError('Insufficient purged peer rows')
            model, fit = train_context(x, y, train, val, cfg, di); del model
            model, _ = train_context(x, y, refit, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            out = np.full(len(y), np.nan, np.float32); out[pred] = predict(model, x, np.flatnonzero(pred))
            if not np.array_equal(np.isfinite(out), pred): raise AssertionError('Invalid peer predictions')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt'))
            np.save(path.with_suffix('.pred.npy'), out)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(refit.sum()),
                       last_inner_train_label=str(max(dates[di[train] + 10])), first_inner_validation_signal=str(min(dates[di[val]])),
                       last_refit_label=str(max(dates[di[refit] + 10])), first_execution=ix['dates'][start],
                       last_execution=ix['dates'][end - 1], seconds=round(time.monotonic() - started, 2))
            if row['last_inner_train_label'] >= row['first_inner_validation_signal'] or row['last_refit_label'] >= row['first_execution']:
                raise AssertionError('Peer labels crossed partition')
            atomic_json(path, row); completed.append(dict(model=cfg['id'], month=fold['month'])); del model
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != spec['planned_fits'] or len(completed) != 108: raise AssertionError('Incomplete peer forecasts')
    atomic_json(DEST / 'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=len(completed), newly_fitted=108, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, fits=108)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

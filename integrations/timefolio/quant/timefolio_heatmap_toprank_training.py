"""Matched NDCG@12-weighted learning on sector and absolute H10 targets."""
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
from quant.timefolio_heatmap_absolute_training import DEST as ABS_TRAIN, DATA_ROOT, inputs
from quant.timefolio_heatmap_corrected_training import DEST as SECTOR_TRAIN, CONFIGS as BASE_CONFIGS
from quant.timefolio_heatmap_rank_training import date_groups
from quant.timefolio_heatmap_rank_evaluation import CASES
from quant.timefolio_heatmap_toprank import top_rank_loss
from quant.timefolio_heatmap_walkforward import SOURCE, AlternativeNet, continuous_targets, monthly_folds, fold_masks, daily_ic, predict

DEST = SOURCE.with_name('20260929_toprank_training_v1')
CONFIGS = [dict(c, id=f"{c['architecture']}_toprank_{target}_h10_seed{c['seed']}",
                target=target, objective='toprank', top_k=12)
           for target in ['sector', 'absolute'] for c in BASE_CONFIGS]
EVALUATION_PLAN = dict(
    cases=CASES, buffer_multipliers=[0, 1, 2], accounts=984, new_accounts=576,
    new_toprank_accounts=384, new_untrained_accounts=192, legacy_accounts=408,
    prior_hypotheses=5344, new_hypotheses=1088, joint_hypotheses=6432,
    streams='12 new monthly models +4 equal-rank ensembles;17 legacy absolute/sector/online streams;8 pre-frozen untrained streams.41x24 accounts.',
    comparisons='Every top-rank CNN versus cash, same-target matched MLP, same-target uniform-pairwise CNN, matched untrained CNN, and fixed online_nonimage. Positive buffers also versus its own buffer0.',
    execution='Unchanged target5%,rebalance5,NAVband5bp,gross80%,sector min(statutory,20%),smallcap30%,dated stock caps,orders3/10,completed-week turnover.',
    gate='Only fixed CNN ensembles qualify. Same-target ensemble and all3seeds each require positive return,MDD>-20%,>=2positive quarters,<4turnover failures. Every comparator/both blocks adjustedp<0.025 and simultaneous_lower95>0; stress and fresh confirmation still required.',
    bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
    interpretation='Adaptive development on reused dates. Retain all5344 hypotheses; shorter news147 remains separate alpha0.025. No new reserved outcomes. Original untrained signs and all seeds fixed before account outcomes.')


def reference(cfg):
    member = f"seed{cfg['seed']}"
    if cfg['target'] == 'sector':
        matched = next(c for c in BASE_CONFIGS if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])
        return SECTOR_TRAIN, matched['id']
    return ABS_TRAIN, f"{cfg['architecture']}_absolute_rank_h10_{member}"


def train_toprank(x, y, train, val, cfg, di, ci, *, epochs=None):
    """One date per optimizer step; stable security keys and purged IC selection."""
    if cfg['objective'] != 'toprank' or cfg['top_k'] != 12: raise ValueError('Frozen top-rank objective required')
    if len(ci) != len(y) or len(di) != len(y): raise ValueError('Matching security and date vectors required')
    if np.any(train & val): raise ValueError('Training and validation overlap')
    seed = cfg['seed']; torch.manual_seed(seed)
    model = AlternativeNet(cfg['architecture'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    keys = torch.as_tensor(ci, dtype=torch.int64)
    groups = date_groups(di, train); va = np.flatnonzero(val)
    if not groups or max(map(len, groups)) > 512: raise ValueError('Missing or oversized date group')
    if not np.isfinite(y[train | val]).all(): raise ValueError('Non-finite label')
    best = -np.inf; state = None; best_epoch = 0; history = []
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses = []
        for group in np.random.default_rng(seed + epoch).permutation(len(groups)):
            ids = groups[group]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1))
            loss = top_rank_loss(out, target[ids], top_k=cfg['top_k'], security_keys=keys[ids])
            if not torch.isfinite(loss): raise AssertionError('Non-finite top-rank loss')
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
    parents = {name: json.loads((root / 'protocol.json').read_text()) for name, root in [('sector', SECTOR_TRAIN), ('absolute', ABS_TRAIN)]}
    for manifest in [parents['sector']['hashes'], parents['absolute']['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text())]:
        for filename, sha in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Top-rank input dependency changed')
    p, ix, ci, di, absolute, allowed = inputs()
    labels = dict(absolute=absolute, sector=continuous_targets(p, ci, di, 10, 'sector'))
    if not np.array_equal(absolute, np.load(ABS_TRAIN / 'labels_h10_absolute.npy'), equal_nan=True): raise AssertionError('Absolute labels changed')
    files = [Path(__file__), ABS_TRAIN / 'protocol.json', SECTOR_TRAIN / 'protocol.json', DATA_ROOT / 'data_hashes.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['toprank', 'toprank_evaluation', 'absolute_training', 'corrected_training', 'rank_training',
               'walkforward', 'study', 'features', 'data']]
    partitions = []; DEST.mkdir(exist_ok=True)
    for name, y in labels.items():
        path = DEST / f'labels_h10_{name}.npy'
        if path.exists():
            if not np.array_equal(np.load(path), y, equal_nan=True): raise RuntimeError('Top-rank label changed')
        else: np.save(path, y)
        files.append(path)
    for cfg in CONFIGS:
        root, name = reference(cfg); y = labels[cfg['target']]
        for fold in monthly_folds(ix['dates']):
            path = root / 'models' / name / (fold['month'] + '.json'); files.append(path)
            old = json.loads(path.read_text()); tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if (int(tr.sum()), int(va.sum()), int(rf.sum())) != (old['fit']['training_rows'], old['fit']['validation_rows'], old['refit_rows']):
                raise AssertionError('Top-rank partitions changed')
            partitions.append(dict(model=cfg['id'], month=fold['month'], reference_model=name,
                                   training_rows=int(tr.sum()), validation_rows=int(va.sum()), refit_rows=int(rf.sum()), prediction_rows=int(pr.sum())))
    spec = dict(configs=CONFIGS, parents=parents, partitions=partitions, evaluation_plan=EVALUATION_PLAN, planned_new_fits=108,
                labels='Identical corrected H10 sector-residual and absolute targets/eligibility to each uniform-pairwise parent.',
                loss='Within-date target midranks->grades0..4;detached absolute NDCG@12 swap weights;normalised weighted RankNet;strict label pairs only;fixed security-index keys resolve score ties. Research adaptation,not portfolio objective.',
                training='Same raw1x32x65,AlternativeNet CNN/MLP,seeds17/29/43,AdamWlr0.0007 decay0.001,clip5,max6epochs,purged past20session dailyIC selection,restart/refit. Only objective changes within each target.',
                reference='https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf',
                limitations='Adaptive development,static metadata,remaining actions/dividends and approximate execution; no independent confirmation.',
                hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Top-rank training protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, f in enumerate(files):
            if f.suffix == '.py': shutil.copy2(f, folder / (f'{i:03d}_' + f.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    _, ix, ci, di, _, allowed = inputs(); dates = np.asarray(ix['dates'])
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    records = []
    for cfg in CONFIGS:
        y = np.load(DEST / f"labels_h10_{cfg['target']}.npy")
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if any(path.with_suffix(ext).exists() for ext in ['.json', '.pt', '.pred.npy']): raise RuntimeError('Do not overwrite a top-rank fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if tr.sum() < 1000 or va.sum() < 200: raise RuntimeError('Insufficient purged observations')
            started = time.monotonic(); model, fit = train_toprank(x, y, tr, va, cfg, di, ci); del model
            model, _ = train_toprank(x, y, rf, np.zeros(len(y), bool), cfg, di, ci, epochs=fit['best_epoch'])
            out = np.full(len(y), np.nan, np.float32); out[pr] = predict(model, x, np.flatnonzero(pr))
            if not np.array_equal(np.isfinite(out), pr): raise AssertionError('Top-rank forecast coverage differs')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt')); del model
            np.save(path.with_suffix('.pred.npy'), out)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()),
                       last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                       last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                       last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            root, name = reference(cfg); old = json.loads((root / 'models' / name / path.name).read_text())
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != old[key]: raise AssertionError('Top-rank boundary changed')
            atomic_json(path, row); records.append(dict(model=cfg['id'], month=fold['month']))
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 108 or len(records) != spec['planned_new_fits']: raise AssertionError('Incomplete top-rank fits')
    atomic_json(DEST / 'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=108, newly_fitted=108, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=108)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

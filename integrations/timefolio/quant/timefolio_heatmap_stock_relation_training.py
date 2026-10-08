"""Purged monthly ranking with full same-date feature context."""
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

from quant.timefolio_heatmap_absolute_training import DEST as ABS_TRAIN, DATA_ROOT, inputs
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_rank_training import pairwise_loss
from quant.timefolio_heatmap_stock_relation import StockRelationNet, date_contexts, predict_by_date
from quant.timefolio_heatmap_walkforward import SOURCE, monthly_folds, fold_masks, daily_ic

DEST = SOURCE.with_name('20260929_stock_relation_training_v1')
PRIOR = SOURCE.with_name('20260929_adjustment_band_v1')
READINESS = SOURCE.with_name('20260929_stock_relation_readiness_v1')
PLAN_RECORD = Path(__file__).resolve().parents[1] / '_workspace/timefolio_heatmap_walkforward_v3/stock_relation_next_plan.json'
CONFIGS = [dict(id=f'{arch}_relation_{relation}_h10_seed{seed}', architecture=arch, relation=relation,
                target='absolute', horizon=10, objective='pairwise', seed=seed, lr=.0007, epochs=6)
           for arch in ['cnn', 'mlp'] for relation in ['self', 'mean', 'attention'] for seed in [17, 29, 43]]


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_hashes(manifest):
    for path, sha in manifest.items():
        if digest(path) != sha: raise AssertionError('Stock-relation dependency changed: ' + str(path))


def reference(cfg):
    return f"{cfg['architecture']}_absolute_rank_h10_seed{cfg['seed']}"


def train_relation(x, y, train, val, cfg, di, *, epochs=None):
    """A loss mask selects outputs; it never selects the date's input peers."""
    y, di, train, val = map(np.asarray, [y, di, train, val])
    if (y.ndim != 1 or di.shape != y.shape or train.shape != y.shape or val.shape != y.shape
            or train.dtype != bool or val.dtype != bool or len(x) != len(y) or x.dtype != torch.uint8):
        raise ValueError('Archived images and equally sized labels, dates and boolean masks required')
    if not np.isfinite(y[train | val]).all(): raise ValueError('Selected labels must be finite')
    if np.intersect1d(di[train], di[val]).size: raise ValueError('Training and validation dates overlap')
    if cfg['objective'] != 'pairwise': raise ValueError('Registered pairwise objective required')
    count = cfg['epochs'] if epochs is None else epochs
    if not isinstance(count, int) or isinstance(count, bool) or count < 1: raise ValueError('Positive integer epoch count required')
    groups = date_contexts(di, train)
    if not groups: raise ValueError('Missing supervised training dates')
    seed = cfg['seed']; torch.manual_seed(seed)
    model = StockRelationNet(cfg['architecture'], cfg['relation'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32); va = np.flatnonzero(val)
    best = -np.inf; state = None; best_epoch = 0; history = []
    for epoch in range(count):
        model.train(); losses = []
        for group in np.random.default_rng(seed + epoch).permutation(len(groups)):
            context, supervised = groups[group]; opt.zero_grad(set_to_none=True)
            scores = model(x[context].float().div(127.5).sub(1))
            loss = pairwise_loss(scores[supervised], target[context[supervised]])
            if not torch.isfinite(loss): raise AssertionError('Non-finite relation loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); opt.step()
            losses.append(float(loss.detach()))
        metric = 0.
        if len(va):
            values = np.full(len(y), np.nan); values[va] = predict_by_date(model, x, di, va)
            metric = daily_ic(y, values, val, di)
            if not np.isfinite(metric): raise AssertionError('Non-finite relation validation metric')
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), inner_ic=metric))
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch + 1
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, history=history, training_rows=int(train.sum()),
                       validation_rows=int(val.sum()), training_dates=len(groups),
                       training_context_rows=sum(len(c) for c, _ in groups),
                       validation_context_rows=sum(len(c) for c, _ in date_contexts(di, val)))


def parent_ready():
    validation = json.loads((PRIOR / 'validation_complete.json').read_text())
    summary = json.loads((PRIOR / 'evaluation_summary.json').read_text())
    if validation['status'] != 'numerical_validation_complete' or validation['portfolios'] != 1824:
        raise RuntimeError('Finish parent numerical validation')
    if summary['portfolios'] != 1824 or summary['joint_hypotheses'] != 11168:
        raise RuntimeError('Unexpected parent evaluation')
    if summary['robust_candidate_gate_passed']:
        raise RuntimeError('Prioritize parent candidate stress and fresh confirmation')


def freeze():
    parent_ready(); plan = json.loads(PLAN_RECORD.read_text())
    expected = dict(architectures=['cnn', 'mlp'], relations=['self', 'mean', 'attention'], target='absolute',
                    horizon=10, seeds=[17, 29, 43], monthly_fits=162, total_accounts=744,
                    new_accounts=288, legacy_accounts=456, prior_hypotheses=11168, joint_hypotheses=11744)
    for key, value in expected.items():
        if plan[key] != value: raise AssertionError('Registered relation plan changed: ' + key)
    readiness = json.loads((READINESS / 'readiness.json').read_text())
    if json.loads((READINESS / 'main_review_complete.json').read_text())['status'] != 'synthetic_readiness_reviewed':
        raise RuntimeError('Review primitive readiness first')
    parent = json.loads((ABS_TRAIN / 'protocol.json').read_text())
    for manifest in [plan['input_hashes'], readiness['hashes'], parent['hashes'],
                     json.loads((DATA_ROOT / 'protocol.json').read_text())['hashes'],
                     json.loads((DATA_ROOT / 'data_hashes.json').read_text())]: check_hashes(manifest)
    _, ix, _, di, y, allowed = inputs()
    if not np.array_equal(y, np.load(ABS_TRAIN / 'labels_h10_absolute.npy'), equal_nan=True):
        raise AssertionError('Relation target changed')
    files = [Path(__file__), PLAN_RECORD, READINESS / 'readiness.json', READINESS / 'main_review_complete.json',
             ABS_TRAIN / 'protocol.json', ABS_TRAIN / 'labels_h10_absolute.npy', DATA_ROOT / 'protocol.json',
             DATA_ROOT / 'data_hashes.json', PRIOR / 'validation_complete.json', PRIOR / 'evaluation_summary.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['stock_relation', 'stock_relation_evaluation', 'absolute_training', 'rank_training',
               'walkforward', 'study', 'features', 'data']]
    files += [Path(__file__).resolve().parents[1] / 'tests' / ('test_timefolio_' + name + '.py')
              for name in ['stock_relation', 'stock_relation_training', 'stock_relation_evaluation']]
    partitions = []
    for cfg in CONFIGS:
        for fold in monthly_folds(ix['dates']):
            path = ABS_TRAIN / 'models' / reference(cfg) / (fold['month'] + '.json'); files.append(path)
            old = json.loads(path.read_text()); tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if (int(tr.sum()), int(va.sum()), int(rf.sum())) != (old['fit']['training_rows'], old['fit']['validation_rows'], old['refit_rows']):
                raise AssertionError('Relation partitions changed')
            partitions.append(dict(model=cfg['id'], month=fold['month'], reference_model=reference(cfg),
                                   training_rows=int(tr.sum()), validation_rows=int(va.sum()),
                                   refit_rows=int(rf.sum()), prediction_rows=int(pr.sum()),
                                   training_context_rows=sum(len(c) for c, _ in date_contexts(di, tr)),
                                   validation_context_rows=sum(len(c) for c, _ in date_contexts(di, va)),
                                   refit_context_rows=sum(len(c) for c, _ in date_contexts(di, rf))))
    if len(partitions) != 162: raise AssertionError('Incomplete relation fit grid')
    spec = dict(configs=CONFIGS, plan=plan, partitions=partitions, planned_new_fits=162,
                actions=parent['sector_training']['data_protocol']['parent']['training_parent']['actions'],
                labels_path=str(ABS_TRAIN / 'labels_h10_absolute.npy'),
                context='All sample rows of a signal date; masks affect loss/IC only. No mixed-date batches.',
                hashes={str(path.resolve()): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen relation training changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(dict.fromkeys(files)):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    preflight = json.loads((DEST / 'input_audit.json').read_text())
    if preflight['partitions'] != 162 or not preflight['full_context_verified']:
        raise RuntimeError('Complete relation input audit first')
    _, ix, _, di, y, allowed = inputs(); dates = np.asarray(ix['dates'])
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    records = []
    for cfg in CONFIGS:
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if any(path.with_suffix(ext).exists() for ext in ['.json', '.pt', '.pred.npy']):
                raise RuntimeError('Do not overwrite a stock-relation fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if tr.sum() < 1000 or va.sum() < 200: raise RuntimeError('Insufficient purged observations')
            started = time.monotonic(); model, fit = train_relation(x, y, tr, va, cfg, di); del model
            model, refit = train_relation(x, y, rf, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            forecast = np.full(len(y), np.nan, np.float32)
            forecast[pr] = predict_by_date(model, x, di, np.flatnonzero(pr))
            if not np.array_equal(np.isfinite(forecast), pr): raise AssertionError('Relation forecast coverage changed')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt')); del model
            np.save(path.with_suffix('.pred.npy'), forecast)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()), refit_context_rows=refit['training_context_rows'],
                       last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                       last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                       last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            old = json.loads((ABS_TRAIN / 'models' / reference(cfg) / path.name).read_text())
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != old[key]: raise AssertionError('Relation fold boundary changed')
            atomic_json(path, row); records.append(dict(model=cfg['id'], month=fold['month']))
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 162 or len(records) != spec['planned_new_fits']: raise AssertionError('Incomplete relation fits')
    check_hashes(spec['hashes'])
    atomic_json(DEST / 'forecast_hashes.json', {str(path): digest(path) for path in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=162, newly_fitted=162, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=162)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

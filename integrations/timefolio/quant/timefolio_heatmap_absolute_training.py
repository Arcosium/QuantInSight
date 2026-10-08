"""Matched H10 absolute-return ranking with corrected inputs and three seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_corrected_features import DEST as DATA_ROOT
from quant.timefolio_heatmap_corrected_training import DEST as SECTOR_TRAIN, CONFIGS as SECTOR_CONFIGS
from quant.timefolio_heatmap_rank_training import DEST as LABEL_ROOT, train_daywise
from quant.timefolio_heatmap_rank_evaluation import CASES
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE, continuous_targets, monthly_folds, fold_masks, predict

DEST = SOURCE.with_name('20260929_absolute_training_v1')
CONFIGS = [dict(c, id=f"{c['architecture']}_absolute_rank_h10_seed{c['seed']}", target='absolute')
           for c in SECTOR_CONFIGS]
EVALUATION_PLAN = dict(
    cases=CASES, buffer_multipliers=[0, 1, 2], accounts=408, new_accounts=192,
    legacy_accounts=216, prior_hypotheses=4896, new_hypotheses=448, joint_hypotheses=5344,
    streams='6 new forecasts and2 equal-rank ensembles;8 corrected sector CNN/MLP members and legacy online_nonimage.17streams x8cases x3buffers.',
    comparisons='Every absolute CNN vs cash, matched absolute MLP, matched corrected sector CNN and fixed online_nonimage; positive buffers also vs same absolute CNN buffer0.',
    execution='Unchanged weight5%,rebalance5,NAVband5bp,gross80%,sector min(statutory,20%),smallcap30%,dated stock caps,orders3/10,completed-week turnover.',
    gate='Only fixed CNN ensemble may qualify. Ensemble and all3seeds must each have positive return,MDD>-20%,>=2positive quarters,<4turnover failures. All comparators/both blocks require adjustedp<0.025 and simultaneous_lower95>0. Stress and fresh confirmation still required.',
    bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
    interpretation='Adaptive development on already used dates. Retain all4896 earlier full-period hypotheses. Separate shorter news147 family keeps alpha0.025. No fresh reserved outcomes.')


def reference_config(cfg):
    return next(c for c in SECTOR_CONFIGS if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])


def inputs():
    p, ix, ci, di = context(DATA_ROOT)
    if ix['dates'][-1] != '20260923':
        raise AssertionError('Absolute experiment date boundary changed')
    p['factor'] = np.cumprod(p['split'], axis=1)
    sector = continuous_targets(p, ci, di, 10, 'sector')
    expected = np.load(LABEL_ROOT / 'labels_h10.npy')
    if not np.array_equal(sector, expected, equal_nan=True):
        raise AssertionError('Matched sector labels changed')
    absolute = continuous_targets(p, ci, di, 10, 'absolute')
    if not np.array_equal(np.isfinite(absolute), np.isfinite(sector)):
        raise AssertionError('Absolute target changed the observation set')
    allowed = np.zeros(len(ci), bool); exists = di + 1 < len(ix['dates'])
    allowed[exists] = p['trade_allowed'][ci[exists], di[exists] + 1].astype(bool)
    return p, ix, ci, di, absolute, allowed


def freeze():
    parent = json.loads((SECTOR_TRAIN / 'protocol.json').read_text())
    data = json.loads((DATA_ROOT / 'protocol.json').read_text())
    for manifest in [data['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text()), parent['hashes']]:
        for filename, digest in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise AssertionError('Absolute experiment dependency changed')
    _, ix, _, di, y, allowed = inputs(); partitions = []
    files = [Path(__file__), DATA_ROOT / 'protocol.json', DATA_ROOT / 'data_hashes.json',
             SECTOR_TRAIN / 'protocol.json', LABEL_ROOT / 'labels_h10.npy']
    for cfg in CONFIGS:
        reference = reference_config(cfg)
        if {k: v for k, v in cfg.items() if k not in ['id', 'target']} != {k: v for k, v in reference.items() if k not in ['id', 'target']}:
            raise AssertionError('Absolute experiment differs beyond target')
        for fold in monthly_folds(ix['dates']):
            path = SECTOR_TRAIN / 'models' / reference['id'] / (fold['month'] + '.json')
            files.append(path); old = json.loads(path.read_text())
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if int(tr.sum()) != old['fit']['training_rows'] or int(va.sum()) != old['fit']['validation_rows'] or int(rf.sum()) != old['refit_rows']:
                raise AssertionError('Absolute experiment partitions changed')
            partitions.append(dict(model=cfg['id'], month=fold['month'], reference_model=reference['id'],
                                   training_rows=int(tr.sum()), validation_rows=int(va.sum()),
                                   refit_rows=int(rf.sum()), prediction_rows=int(pr.sum())))
    DEST.mkdir(exist_ok=True); label_path = DEST / 'labels_h10_absolute.npy'
    if label_path.exists():
        if not np.array_equal(np.load(label_path), y, equal_nan=True): raise RuntimeError('Absolute labels changed')
    else: np.save(label_path, y)
    files.append(label_path)
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['absolute_evaluation', 'corrected_training', 'corrected_features', 'rank_training',
               'rank_evaluation', 'walkforward', 'study', 'features', 'data']]
    spec = dict(configs=CONFIGS, data_protocol=data, sector_training=parent, partitions=partitions,
                evaluation_plan=EVALUATION_PLAN, planned_new_fits=54,
                labels='Same corrected execution-entry to H10 close return; subtract common0.004,divide0.10,clip[-3,3]. No sector median subtraction. Common cost shift preserves ranking except saturation/rounding ties; this is not a calibrated net-return forecast or a cash gate.',
                training='Same raw one-channel32x65 images,AlternativeNet CNN/MLP,seed17/29/43,date-balanced pairwise loss,AdamWlr0.0007 decay0.001,clip5,max6epochs with purged past20session dailyIC selection;restart/refit all observed labels.Only target changes; all54fits new.',
                ensemble='Fixed equal mean of within-date eligible percentile ranks for all3seeds. No best-seed selection.',
                scope='Sector caps allow sector allocations; test whether subtracting sector returns discards usable differences. This rationale is not evidence of an edge. Frozen input/action/execution/metadata limitations remain.',
                hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Absolute training protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, f in enumerate(files):
            if f.suffix == '.py': shutil.copy2(f, folder / (f'{i:03d}_' + f.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    _, ix, _, di, y, allowed = inputs(); dates = np.asarray(ix['dates'])
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    records = []
    for cfg in CONFIGS:
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if any(path.with_suffix(ext).exists() for ext in ['.json', '.pt', '.pred.npy']):
                raise RuntimeError('Do not overwrite an absolute fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if tr.sum() < 1000 or va.sum() < 200: raise RuntimeError('Insufficient purged observations')
            started = time.monotonic(); model, fit = train_daywise(x, y, tr, va, cfg, di); del model
            model, _ = train_daywise(x, y, rf, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            forecast = np.full(len(y), np.nan, np.float32); forecast[pr] = predict(model, x, np.flatnonzero(pr))
            if not np.array_equal(np.isfinite(forecast), pr): raise AssertionError('Absolute forecast coverage differs')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt')); del model
            np.save(path.with_suffix('.pred.npy'), forecast)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()),
                       last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                       last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                       last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            old = json.loads((SECTOR_TRAIN / 'models' / reference_config(cfg)['id'] / path.name).read_text())
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != old[key]: raise AssertionError('Absolute fold boundary changed')
            atomic_json(path, row); records.append(dict(model=cfg['id'], month=fold['month']))
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 54 or len(records) != spec['planned_new_fits']: raise AssertionError('Incomplete absolute fits')
    atomic_json(DEST / 'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=54, newly_fitted=54, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=54)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

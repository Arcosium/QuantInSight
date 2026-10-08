"""Retrain only folds whose training/selection images actually changed."""
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
from quant.timefolio_heatmap_rank_training import DEST as TRAIN_ROOT, CONFIGS as BASE_CONFIGS, train_daywise
from quant.timefolio_heatmap_rank_seeds import DEST as SEED_ROOT, CONFIGS as SEED_CONFIGS
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT, AlternativeNet, continuous_targets, monthly_folds, fold_masks, predict

DEST = SOURCE.with_name('20260929_corrected_training_v1')
CONFIGS = [c for c in BASE_CONFIGS if c['objective'] == 'pairwise' and c['horizon'] == 10] + SEED_CONFIGS


def previous_folder(cfg):
    return (TRAIN_ROOT if cfg['seed'] == 17 else SEED_ROOT) / 'models' / cfg['id']


def inputs():
    p, ix, ci, di = context(DATA_ROOT); original = np.load(SOURCE / 'samples.npz')
    if not np.array_equal(ci, original['ci']) or not np.array_equal(di, original['di']):
        raise AssertionError('Sample universe changed; reuse needs a new plan')
    p['factor'] = np.cumprod(p['split'], axis=1)
    y = continuous_targets(p, ci, di, 10, 'sector')
    if not np.array_equal(y, np.load(TRAIN_ROOT / 'labels_h10.npy'), equal_nan=True):
        raise AssertionError('Corrected labels differ from existing amended H10 labels')
    allowed = np.zeros(len(ci), bool); valid = di + 1 < len(ix['dates'])
    allowed[valid] = p['trade_allowed'][ci[valid], di[valid] + 1].astype(bool)
    old = np.load(MODEL_ROOT / 'images_raw.npy', mmap_mode='r'); new = np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')
    if old.shape != new.shape: raise AssertionError('Image shape changed')
    changed = np.any(old != new, axis=(1, 2)); plan = []
    for cfg in CONFIGS:
        for fold in monthly_folds(ix['dates']):
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            prior = json.loads((previous_folder(cfg) / (fold['month'] + '.json')).read_text())
            if cfg != prior['config'] or prior['fit']['training_rows'] != int(tr.sum()) or prior['fit']['validation_rows'] != int(va.sum()) or prior['refit_rows'] != int(rf.sum()):
                raise AssertionError('Existing fit configuration or partition changed')
            touched = int((changed & (tr | va | rf)).sum())
            plan.append(dict(model=cfg['id'], month=fold['month'], refit_needed=touched > 0,
                             changed_fit_or_selection_rows=touched, changed_prediction_rows=int((changed & pr).sum())))
    return p, ix, ci, di, y, allowed, changed, plan


def freeze():
    data_protocol = json.loads((DATA_ROOT / 'protocol.json').read_text())
    manifests = [data_protocol['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text())]
    for manifest in manifests:
        for filename, value in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value: raise AssertionError('Corrected data input changed')
    *_, changed, plan = inputs()
    files = [Path(__file__), DATA_ROOT / 'protocol.json', DATA_ROOT / 'data_hashes.json', DATA_ROOT / 'image_audit.json',
             TRAIN_ROOT / 'labels_h10.npy', TRAIN_ROOT / 'protocol.json', SEED_ROOT / 'protocol.json']
    for cfg in CONFIGS:
        folder = previous_folder(cfg)
        files += sorted(folder.glob('*.json')) + sorted(folder.glob('*.pt')) + sorted(folder.glob('*.pred.npy'))
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['corrected_features', 'rank_training', 'rank_seeds', 'walkforward', 'study', 'features', 'data']]
    spec = dict(configs=CONFIGS, data_protocol=data_protocol, plan=plan, changed_images=int(changed.sum()),
                policy='All6 H10 pairwise CNN/MLP seed17/29/43 configs retained. Same amended labels, dates, optimizer, architecture, seed and epoch-selection rule. Reuse a checkpoint only if every train/validation/refit image is unchanged; otherwise fully retrain and refit.',
                inference='Reused folds with changed prediction images are reinferred. Unchanged-image predictions must still match old scores within1e-6. Wholly unchanged predictions are copied exactly. No selection by performance.',
                planned_fits=sum(r['refit_needed'] for r in plan), planned_reused_folds=sum(not r['refit_needed'] for r in plan),
                evaluation='Preserve all three seeds and fixed equal-rank ensemble. Same8cases and buffers0/N/2N, same dated limits/costs/turnover. Compare cash, matched corrected MLP and fixed original online_nonimage; positive-buffer ensemble also must beat its corrected buffer0. Retain3552 previous hypotheses, add352 for3904 total. Alpha0.025; blocks5/10,4000draws,seed57; no independent-confirmation claim.',
                gate='Only ensemble can qualify; all3comparators/bothblocks plus same-corrected-CNN-buffer0 for positive buffers must pass adjustedp<0.025 and lower95>0. Ensemble and3seeds must each be profitable, MDD>-20%, at least2positivequarters and fewer than4turnover failures. Stress/fresh confirmation still required.',
                scope='Data correction, not a claim that54 slightly changed images explain prior seed instability. The legacy online_nonimage comparator stays fixed; matched MLPs receive the same corrected inputs. No reserved outcomes.',
                hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Corrected training protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, f in enumerate(files):
            if f.suffix == '.py': shutil.copy2(f, folder / (f'{i:03d}_' + f.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    p, ix, ci, di, y, allowed, changed, plan = inputs(); dates = np.asarray(ix['dates'])
    if plan != spec['plan']: raise AssertionError('Reuse plan changed')
    planned = {(r['model'], r['month']): r for r in plan}
    images = np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')
    x = torch.from_numpy(np.array(images[:, None], copy=True)); records = []
    for cfg in CONFIGS:
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            decision = planned[(cfg['id'], fold['month'])]; output = folder / (fold['month'] + '.json')
            prior_path = previous_folder(cfg) / output.name; prior = json.loads(prior_path.read_text())
            if output.exists(): raise RuntimeError('Do not overwrite a completed fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            started = time.monotonic(); old_prediction = np.load(prior_path.with_suffix('.pred.npy'))
            if not np.array_equal(np.isfinite(old_prediction), pr): raise AssertionError('Prior forecast coverage differs')
            if decision['refit_needed']:
                model, fit = train_daywise(x, y, tr, va, cfg, di); del model
                model, _ = train_daywise(x, y, rf, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
                forecast = np.full(len(y), np.nan, np.float32); forecast[pr] = predict(model, x, np.flatnonzero(pr))
                torch.save(dict(config=cfg, state_dict=model.state_dict()), output.with_suffix('.pt')); del model
            else:
                if (changed & (tr | va | rf)).any(): raise AssertionError('Cannot reuse touched fit')
                fit = prior['fit']; shutil.copy2(prior_path.with_suffix('.pt'), output.with_suffix('.pt'))
                if decision['changed_prediction_rows']:
                    saved = torch.load(output.with_suffix('.pt'), map_location='cpu', weights_only=True)
                    if saved['config'] != cfg: raise AssertionError('Reused checkpoint config changed')
                    model = AlternativeNet(cfg['architecture']); model.load_state_dict(saved['state_dict'])
                    forecast = np.full(len(y), np.nan, np.float32); forecast[pr] = predict(model, x, np.flatnonzero(pr)); del model
                    error = float(np.max(np.abs(forecast[pr & ~changed] - old_prediction[pr & ~changed])))
                    if error > 1e-6: raise AssertionError('Unchanged image predictions changed under reused weights')
                else: forecast = old_prediction.copy()
            if not np.array_equal(np.isfinite(forecast), pr): raise AssertionError('New prediction coverage differs')
            np.save(output.with_suffix('.pred.npy'), forecast)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()),
                       last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                       last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                       last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != prior[key]: raise AssertionError('Partition boundary changed')
            atomic_json(output, row)
            record = dict(**decision, seconds=row['seconds'], forecast_values_changed=int(np.count_nonzero(forecast[pr] != old_prediction[pr])))
            if not decision['refit_needed']:
                record['checkpoint_bytes_equal'] = output.with_suffix('.pt').read_bytes() == prior_path.with_suffix('.pt').read_bytes()
                if not record['checkpoint_bytes_equal']: raise AssertionError('Reused weights changed')
            records.append(record); print(json.dumps(record), flush=True)
    atomic_json(DEST / 'reuse_audit.json', records)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 54: raise AssertionError('Corrected forecast set incomplete')
    atomic_json(DEST / 'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=len(records), newly_fitted=sum(r['refit_needed'] for r in records),
                weights_reused=sum(not r['refit_needed'] for r in records), evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=len(records))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); a = ap.parse_args()
    if a.action == 'freeze': freeze()
    else: run()

"""Matched absolute-H10 ranking on historical daily-summary heatmaps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from quant.timefolio_heatmap_absolute_training import DEST as SNAPSHOT_TRAIN, CONFIGS as BASE_CONFIGS, inputs
from quant.timefolio_heatmap_sector_ceiling import DEST as PRIOR
from quant.timefolio_heatmap_stock_relation_training import DATA_ROOT, digest, check_hashes
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_rank_training import train_daywise
from quant.timefolio_heatmap_walkforward import monthly_folds, fold_masks, predict

DEST = PRIOR.with_name('20260929_daily_history_training_v1')
FEATURES = PRIOR.with_name('20260929_daily_history_features_v1')
PLAN_RECORD = Path(__file__).resolve().parents[1] / '_workspace/timefolio_heatmap_walkforward_v3/daily_history_next_plan.json'
ENCODINGS = {'history': 'history', 'reverse': 'history_reverse'}
CONFIGS = [dict(cfg, id=f"{cfg['architecture']}_daily_{token}_h10_seed{cfg['seed']}", encoding=encoding)
           for token, encoding in ENCODINGS.items() for cfg in BASE_CONFIGS]
EVALUATION_PLAN = dict(accounts=592, new_accounts=448, legacy_accounts=144, streams=37,
    prior_hypotheses=12416, new_hypotheses=832, joint_hypotheses=13248, new_fits=108,
    sector_modes=['research20', 'formula'], buffers=[12, 24], orders=[3, 10], refresh=[1, 5],
    weight=.05, top_n=12, gross=.8, rebalance=5, band=.0005,
    comparisons='Cash, matched representation MLP, matched representation untrained CNN, original snapshot CNN, other daily representation CNN, online_nonimage; formula mode also versus its own research20 account.',
    gate='Only fixed CNN ensembles; ensemble and all3seeds positive return,MDD>-20%,>=2positive quarters,<4turnover failures. Every comparator at both blocks adjustedp<.025 and simultaneous_lower95>0. Stress and fresh confirmation still required.',
    bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
    news_family=dict(hypotheses=147, alpha=.025, separate=True))


def reference_config(cfg):
    return next(c for c in BASE_CONFIGS if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])


def feature_path(cfg):
    if cfg['encoding'] not in ENCODINGS.values(): raise ValueError('Registered daily-history representation required')
    return FEATURES / ('images_' + cfg['encoding'] + '.npy')


def parent_ready():
    review = json.loads((PRIOR / 'validation_complete.json').read_text())
    summary = json.loads((PRIOR / 'evaluation_summary.json').read_text())
    if review['status'] != 'numerical_validation_complete' or review['portfolios'] != 656:
        raise RuntimeError('Finish the main sector-ceiling review first')
    if (summary['portfolios'], summary['joint_hypotheses']) != (656, 12416):
        raise RuntimeError('Unexpected sector-ceiling parent')
    if review['candidates'] or summary['robust_candidate_gate_passed']:
        raise RuntimeError('Prioritize parent candidate stress and confirmation')


def freeze():
    parent_ready(); plan = json.loads(PLAN_RECORD.read_text())
    if plan['configs'] != CONFIGS or plan['evaluation_plan'] != EVALUATION_PLAN:
        raise AssertionError('Registered daily-history experiment changed')
    feature_spec = json.loads((FEATURES / 'protocol.json').read_text())
    review = json.loads((FEATURES / 'main_review.json').read_text())
    if review['status'] != 'main_reviewed_feature_preparation': raise RuntimeError('Feature review required')
    parent = json.loads((PRIOR / 'protocol.json').read_text())
    snapshot = json.loads((SNAPSHOT_TRAIN / 'protocol.json').read_text())
    manifests = [plan['evidence_hashes'], feature_spec['hashes'], json.loads((FEATURES / 'image_hashes.json').read_text()),
                 json.loads((DATA_ROOT / 'data_hashes.json').read_text()), snapshot['hashes']]
    for manifest in manifests: check_hashes(manifest)
    _, ix, _, di, y, allowed = inputs()
    if not np.array_equal(y, np.load(SNAPSHOT_TRAIN / 'labels_h10_absolute.npy'), equal_nan=True):
        raise AssertionError('Matched absolute labels changed')
    files = [Path(__file__), PLAN_RECORD, PRIOR / 'protocol.json', PRIOR / 'validation_complete.json',
             PRIOR / 'evaluation_summary.json', SNAPSHOT_TRAIN / 'protocol.json', SNAPSHOT_TRAIN / 'labels_h10_absolute.npy',
             FEATURES / 'protocol.json', FEATURES / 'feature_audit.json', FEATURES / 'main_review.json',
             FEATURES / 'image_hashes.json', DATA_ROOT / 'data_hashes.json']
    files += [FEATURES / ('images_' + encoding + '.npy') for encoding in ENCODINGS.values()]
    files += [Path(filename) for filename in json.loads((DATA_ROOT / 'data_hashes.json').read_text())]
    partitions = []
    for cfg in CONFIGS:
        ref = reference_config(cfg)
        assert {k: v for k, v in cfg.items() if k not in ['id', 'encoding']} == {k: v for k, v in ref.items() if k != 'id'}
        for fold in monthly_folds(ix['dates']):
            path = SNAPSHOT_TRAIN / 'models' / ref['id'] / (fold['month'] + '.json')
            files.extend([path, path.with_suffix('.pt'), path.with_suffix('.pred.npy')])
            old = json.loads(path.read_text()); tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            assert (int(tr.sum()), int(va.sum()), int(rf.sum())) == (old['fit']['training_rows'], old['fit']['validation_rows'], old['refit_rows'])
            partitions.append(dict(model=cfg['id'], month=fold['month'], reference_model=ref['id'],
                training_rows=int(tr.sum()), validation_rows=int(va.sum()), refit_rows=int(rf.sum()), prediction_rows=int(pr.sum())))
    files += [Path(__file__).with_name('timefolio_heatmap_' + name + '.py') for name in
              ['daily_history', 'daily_history_evaluation', 'absolute_training', 'rank_training', 'walkforward', 'study', 'data']]
    files += [Path(__file__).resolve().parents[1] / 'tests' / ('test_timefolio_daily_history_' + name + '.py')
              for name in ['training', 'evaluation']]
    files = sorted(set(path.resolve() for path in files))
    spec = dict(configs=CONFIGS, evaluation_plan=EVALUATION_PLAN, planned_new_fits=108,
        feature_protocol=feature_spec, partitions=partitions, actions=parent['parent']['training']['actions'],
        training='Same AlternativeNet CNN/MLP, absoluteH10 labels, three seeds, masks, date-balanced pairwise loss, AdamW and past20session epoch selection as the snapshot study; restart/refit. Only the daily-summary representation changes.',
        missing='All46596 rows retained; historical neutral values have no additional missingness channel. This representation limitation is disclosed, not silently filtered.',
        scope='Repeated255-session development; all earlier hypotheses retained. No fresh confirmation or reserved date outcomes.',
        hashes={str(path): digest(path) for path in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Frozen daily-history protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, source in enumerate(files):
            if source.suffix == '.py': shutil.copy2(source, folder / (f'{i:03d}_' + source.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    _, ix, _, di, y, allowed = inputs(); dates = np.asarray(ix['dates'])
    active_encoding, x, records = None, None, []
    for cfg in CONFIGS:
        if active_encoding != cfg['encoding']:
            x = torch.from_numpy(np.array(np.load(feature_path(cfg), mmap_mode='r')[:, None], copy=True))
            active_encoding = cfg['encoding']
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if any(path.with_suffix(ext).exists() for ext in ['.json', '.pt', '.pred.npy']):
                raise RuntimeError('Do not overwrite a daily-history fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if tr.sum() < 1000 or va.sum() < 200: raise RuntimeError('Insufficient purged observations')
            started = time.monotonic(); model, fit = train_daywise(x, y, tr, va, cfg, di); del model
            model, _ = train_daywise(x, y, rf, np.zeros(len(y), bool), cfg, di, epochs=fit['best_epoch'])
            forecast = np.full(len(y), np.nan, np.float32); forecast[pr] = predict(model, x, np.flatnonzero(pr))
            if not np.array_equal(np.isfinite(forecast), pr): raise AssertionError('Daily-history forecast coverage differs')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt')); del model
            np.save(path.with_suffix('.pred.npy'), forecast)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()),
                last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            reference = SNAPSHOT_TRAIN / 'models' / reference_config(cfg)['id'] / path.name
            old = json.loads(reference.read_text())
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != old[key]: raise AssertionError('Daily-history fold boundary changed')
            atomic_json(path, row); records.append(dict(model=cfg['id'], month=fold['month']))
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 108 or len(records) != spec['planned_new_fits']: raise AssertionError('Incomplete daily-history fits')
    check_hashes(spec['hashes'])
    atomic_json(DEST / 'forecast_hashes.json', {str(path): digest(path) for path in paths})
    artifacts = sorted(path for path in (DEST / 'models').glob('*/*') if path.suffix in ['.json', '.npy', '.pt'])
    assert len(artifacts) == 324
    atomic_json(DEST / 'model_artifact_hashes.json', {str(path): digest(path) for path in artifacts})
    atomic_json(DEST / 'training_complete.json', dict(folds=108, newly_fitted=108, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=108)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run'])
    if parser.parse_args().action == 'freeze': freeze()
    else: run()

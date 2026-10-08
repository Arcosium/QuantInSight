"""Feasible-portfolio ranking on matched sector and absolute H10 targets."""
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
from quant.timefolio_heatmap_feasible_rank import feasible_pool, feasible_pair_loss
from quant.timefolio_heatmap_stock_limits import historical_stock_caps
from quant.timefolio_heatmap_toprank_training import DEST as TOP_TRAIN
from quant.timefolio_heatmap_walkforward import SOURCE, AlternativeNet, continuous_targets, monthly_folds, fold_masks, daily_ic, predict

DEST = SOURCE.with_name('20260929_feasible_training_v1')
CONFIGS = [dict(c, id=f"{c['architecture']}_feasible_{target}_h10_seed{c['seed']}",
                target=target, objective='feasible_pair', pool_n=12, random_draws=16, pool_seed=317)
           for target in ['sector', 'absolute'] for c in BASE_CONFIGS]
EVALUATION_PLAN = dict(
    cases=CASES, buffer_multipliers=[0, 1, 2], accounts=1368, new_accounts=384, legacy_accounts=984,
    prior_hypotheses=6432, new_hypotheses=1280, joint_hypotheses=7712,
    streams='12new monthly models+4fixed rank ensembles;41legacy top-rank/uniform/untrained/online streams.57x24accounts.',
    comparisons='Every feasible CNN versus cash, same-target feasible MLP, same-target uniform CNN, same-target top-rank CNN, matched untrained CNN, and fixed online_nonimage. Positive buffers also versus own buffer0.',
    execution='Unchanged target5%,rebalance5,NAVband5bp,gross80%,sector min(statutory,20%),smallcap30%,dated stock caps,orders3/10,completed-week turnover.',
    gate='Only fixed CNN ensembles qualify. Same-target ensemble and all3seeds each need positive return,MDD>-20%,>=2positive quarters,<4turnover failures. Every comparison/both blocks adjustedp<0.025,simultaneous_lower95>0. Stress and fresh confirmation still required.',
    bootstrap=dict(blocks=[5, 10], draws=4000, seed=57, alpha=.025),
    interpretation='Adaptive reused development dates. Retain all6432 full-period hypotheses; shorter news147 remains separatealpha0.025. No reserved outcomes. Fresh-account invested-part surrogate does not optimise holdings, execution, turnover or totalNAV.')

def reference(cfg):
    member = f"seed{cfg['seed']}"
    if cfg['target'] == 'sector':
        matched = next(c for c in BASE_CONFIGS if c['architecture'] == cfg['architecture'] and c['seed'] == cfg['seed'])
        return SECTOR_TRAIN, matched['id']
    return ABS_TRAIN, f"{cfg['architecture']}_absolute_rank_h10_{member}"


def metadata_for_date(p, index, ci, di, ids, stock_caps):
    """Only selected training-date metadata; never read future prices or outcomes."""
    signal_dates = np.unique(di[ids])
    if len(signal_dates) != 1: raise ValueError('One training date required')
    d = int(signal_dates[0]); c = ci[ids]
    if d + 1 >= len(index['dates']): raise ValueError('Missing execution-date policy')
    if not np.all(p['eligible'][c, d]) or not np.all(p['trade_allowed'][c, d + 1]):
        raise ValueError('Pool contains an inadmissible training security')
    return dict(eligible=np.ones(len(ids), bool), sector=p['sector'][c],
                sector_caps=np.minimum(p['sector_cap'][c, d].astype(float), .20),
                market_cap=p['market_cap'][c, d].astype(float), codes=np.asarray(index['codes'])[c],
                stock_caps=stock_caps[c, d + 1], signal_date=int(index['dates'][d]))


def train_feasible(x, y, train, val, cfg, di, ci, p, index, stock_caps, *, epochs=None):
    """One date per step; observed labels generate feasible candidate targets."""
    if (cfg['objective'], cfg['pool_n'], cfg['random_draws'], cfg['pool_seed']) != ('feasible_pair', 12, 16, 317):
        raise ValueError('Frozen feasible-pool objective required')
    if len(ci) != len(y) or len(di) != len(y): raise ValueError('Matching security and date vectors required')
    if np.any(train & val): raise ValueError('Training and validation overlap')
    seed = cfg['seed']; torch.manual_seed(seed)
    model = AlternativeNet(cfg['architecture'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=.001)
    target = torch.as_tensor(np.nan_to_num(y), dtype=torch.float32)
    groups = date_groups(di, train); va = np.flatnonzero(val)
    if not groups or max(map(len, groups)) > 512: raise ValueError('Missing or oversized date group')
    if not np.isfinite(y[train | val]).all(): raise ValueError('Non-finite label')
    metadata = [metadata_for_date(p, index, ci, di, ids, stock_caps) for ids in groups]
    best = -np.inf; state = None; best_epoch = 0; history = []
    for epoch in range(epochs or cfg['epochs']):
        model.train(); losses = []; pool_sizes = []; zero_days = 0
        for group in np.random.default_rng(seed + epoch).permutation(len(groups)):
            ids = groups[group]; opt.zero_grad(set_to_none=True)
            out = model(x[ids].float().div(127.5).sub(1))
            meta = metadata[group]; constraints = {k: v for k, v in meta.items() if k != 'signal_date'}
            pool = feasible_pool(out.detach().cpu().numpy(), target[ids].numpy(), **constraints,
                                 top_n=cfg['pool_n'], weight=.05, gross=.8, random_draws=cfg['random_draws'],
                                 seed=cfg['pool_seed'] + meta['signal_date'])
            loss = feasible_pair_loss(out, target[ids], torch.from_numpy(pool))
            if not torch.isfinite(loss): raise AssertionError('Non-finite feasible-pair loss')
            zero_days += int(float(loss.detach()) == 0.)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            losses.append(float(loss.detach())); pool_sizes.append(len(pool))
        metric = 0.
        if len(va):
            values = np.full(len(y), np.nan); values[va] = predict(model, x, va)
            metric = daily_ic(y, values, val, di)
        history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), inner_ic=metric,
                            mean_pool_size=float(np.mean(pool_sizes)), zero_utility_pair_dates=zero_days))
        if not len(va) or metric > best:
            best, best_epoch = metric, epoch + 1
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, dict(best_epoch=best_epoch, history=history, training_rows=int(train.sum()),
                       validation_rows=int(val.sum()), training_dates=len(groups))


def freeze():
    readiness = json.loads((SOURCE.with_name('20260929_feasible_rank_readiness_v1') / 'readiness.json').read_text())
    for filename, sha in readiness['public_source_hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Feasible primitive changed after readiness checks')
    parents = {name: json.loads((root / 'protocol.json').read_text()) for name, root in [('sector', SECTOR_TRAIN), ('absolute', ABS_TRAIN)]}
    for manifest in [parents['sector']['hashes'], parents['absolute']['hashes'], json.loads((TOP_TRAIN / 'protocol.json').read_text())['hashes'], json.loads((DATA_ROOT / 'data_hashes.json').read_text())]:
        for filename, sha in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha: raise AssertionError('Feasible-ranking input dependency changed')
    p, ix, ci, di, absolute, allowed = inputs()
    labels = dict(absolute=absolute, sector=continuous_targets(p, ci, di, 10, 'sector'))
    if not np.array_equal(absolute, np.load(ABS_TRAIN / 'labels_h10_absolute.npy'), equal_nan=True): raise AssertionError('Absolute labels changed')
    files = [Path(__file__), ABS_TRAIN / 'protocol.json', SECTOR_TRAIN / 'protocol.json', DATA_ROOT / 'data_hashes.json',
             TOP_TRAIN / 'protocol.json', SOURCE.with_name('20260929_feasible_rank_readiness_v1') / 'readiness.json']
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in
              ['feasible_rank', 'feasible_evaluation', 'stock_limits', 'replay', 'toprank_training', 'absolute_training', 'corrected_training', 'rank_training',
               'walkforward', 'study', 'features', 'data']]
    partitions = []; DEST.mkdir(exist_ok=True)
    for name, y in labels.items():
        path = DEST / f'labels_h10_{name}.npy'
        if path.exists():
            if not np.array_equal(np.load(path), y, equal_nan=True): raise RuntimeError('Feasible-ranking label changed')
        else: np.save(path, y)
        files.append(path)
    for cfg in CONFIGS:
        root, name = reference(cfg); y = labels[cfg['target']]
        for fold in monthly_folds(ix['dates']):
            path = root / 'models' / name / (fold['month'] + '.json'); files.append(path)
            old = json.loads(path.read_text()); tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if (int(tr.sum()), int(va.sum()), int(rf.sum())) != (old['fit']['training_rows'], old['fit']['validation_rows'], old['refit_rows']):
                raise AssertionError('Feasible-ranking partitions changed')
            partitions.append(dict(model=cfg['id'], month=fold['month'], reference_model=name,
                                   training_rows=int(tr.sum()), validation_rows=int(va.sum()), refit_rows=int(rf.sum()), prediction_rows=int(pr.sum())))
    spec = dict(configs=CONFIGS, parents=parents, partitions=partitions, evaluation_plan=EVALUATION_PLAN, planned_new_fits=108,
                labels='Identical corrected H10 sector-residual and absolute targets/eligibility to each uniform-pairwise parent.',
                loss='Each training date builds up to18greedy feasible portfolios: current outputs,observed labels,16fixed random orders. Rank invested-part mean utility via all-strict-pair logistic loss; numerical ties omitted; no gradient through pool/labels. Candidate constraints use signal metadata and dated next-execution stock cap. No cash gate or totalNAV objective.',
                training='Same raw1x32x65,AlternativeNet CNN/MLP,seeds17/29/43,AdamWlr0.0007 decay0.001,clip5,max6epochs,purged past20session dailyIC selection,restart/refit. Only objective changes within each target.',
                reference='https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf',
                limitations='Adaptive development,static metadata,remaining actions/dividends and approximate execution; no independent confirmation.',
                hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Feasible-ranking training protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, f in enumerate(files):
            if f.suffix == '.py': shutil.copy2(f, folder / (f'{i:03d}_' + f.name))
    return spec


def run():
    spec = freeze(); torch.set_num_threads(4)
    evaluation = SOURCE.with_name('20260929_feasible_evaluation_v1')
    preflight = json.loads((evaluation / 'preflight.json').read_text())
    if preflight['prior_hypotheses'] != 6432 or preflight['joint_hypotheses'] != 7712:
        raise AssertionError('Complete parent evaluation and comparison preflight before training')
    if json.loads((evaluation / 'protocol.json').read_text())['plan'] != EVALUATION_PLAN:
        raise AssertionError('Feasible evaluation plan was not frozen')
    p, ix, ci, di, _, allowed = inputs(); dates = np.asarray(ix['dates'])
    caps = historical_stock_caps(ix['codes'], ix['dates'])
    x = torch.from_numpy(np.array(np.load(DATA_ROOT / 'images_raw.npy', mmap_mode='r')[:, None], copy=True))
    records = []
    for cfg in CONFIGS:
        y = np.load(DEST / f"labels_h10_{cfg['target']}.npy")
        folder = DEST / 'models' / cfg['id']; folder.mkdir(parents=True, exist_ok=True)
        for fold in monthly_folds(ix['dates']):
            path = folder / (fold['month'] + '.json')
            if any(path.with_suffix(ext).exists() for ext in ['.json', '.pt', '.pred.npy']): raise RuntimeError('Do not overwrite a feasible-ranking fold')
            tr, va, rf, pr, _ = fold_masks(di, y, allowed, 10, fold['start'], fold['end'])
            if tr.sum() < 1000 or va.sum() < 200: raise RuntimeError('Insufficient purged observations')
            started = time.monotonic(); model, fit = train_feasible(x, y, tr, va, cfg, di, ci, p, ix, caps); del model
            model, _ = train_feasible(x, y, rf, np.zeros(len(y), bool), cfg, di, ci, p, ix, caps, epochs=fit['best_epoch'])
            out = np.full(len(y), np.nan, np.float32); out[pr] = predict(model, x, np.flatnonzero(pr))
            if not np.array_equal(np.isfinite(out), pr): raise AssertionError('Feasible-ranking forecast coverage differs')
            torch.save(dict(config=cfg, state_dict=model.state_dict()), path.with_suffix('.pt')); del model
            np.save(path.with_suffix('.pred.npy'), out)
            row = dict(config=cfg, fold=fold, fit=fit, refit_rows=int(rf.sum()),
                       last_inner_train_label=str(max(dates[di[tr] + 10])), first_inner_validation_signal=str(min(dates[di[va]])),
                       last_refit_label=str(max(dates[di[rf] + 10])), first_execution=ix['dates'][fold['start']],
                       last_execution=ix['dates'][fold['end'] - 1], seconds=round(time.monotonic() - started, 2))
            root, name = reference(cfg); old = json.loads((root / 'models' / name / path.name).read_text())
            for key in ['last_inner_train_label', 'first_inner_validation_signal', 'last_refit_label', 'first_execution', 'last_execution']:
                if row[key] != old[key]: raise AssertionError('Feasible-ranking boundary changed')
            atomic_json(path, row); records.append(dict(model=cfg['id'], month=fold['month']))
            print(json.dumps(dict(model=cfg['id'], month=fold['month'], epochs=fit['best_epoch'], seconds=row['seconds'])), flush=True)
    paths = sorted((DEST / 'models').glob('*/*.pred.npy'))
    if len(paths) != 108 or len(records) != spec['planned_new_fits']: raise AssertionError('Incomplete feasible-ranking fits')
    atomic_json(DEST / 'forecast_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    atomic_json(DEST / 'training_complete.json', dict(folds=108, newly_fitted=108, evaluation_complete=False))
    print(json.dumps(dict(training_complete=True, folds=108)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'run']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: run()

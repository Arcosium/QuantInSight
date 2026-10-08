"""Score selection quality only; overlapping H5 labels are not account P&L."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quant.timefolio_cnn_dataset import sha


def midranks(a):
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    return (counts.cumsum() - (counts+1)/2)[inverse]


def date_metrics(scores, returns, keys, *, top_k=10):
    scores, returns, keys = map(np.asarray, [scores, returns, keys])
    if scores.shape != returns.shape or scores.shape != keys.shape or not np.isfinite(scores).all():
        raise ValueError('One complete finite date forecast required')
    order = np.lexsort((keys, -scores)); selected = order[:top_k]
    # Never replace a selected stock whose outcome is missing with a survivor.
    usable = np.isfinite(returns)
    top_valid = len(selected) == top_k and usable[selected].all()
    mean = float(returns[usable].mean()) if usable.any() else None
    top = float(returns[selected].mean()) if top_valid else None
    rho = None
    if usable.sum() >= 2:
        x, y = midranks(scores[usable]), midranks(returns[usable])
        if x.std() > 0 and y.std() > 0:
            rho = float(np.corrcoef(x, y)[0, 1])
    return dict(forecast_members=len(scores), observed_outcomes=int(usable.sum()),
        missing_selected_outcomes=int((~usable[selected]).sum()), topk_raw_h5=top,
        observed_universe_raw_h5=mean, topk_excess_h5=top-mean if top is not None and mean is not None else None,
        rank_ic=rho, selected_keys=keys[selected].tolist())


def evaluate(dataset, results, output):
    dataset, results, output = map(Path, [dataset, results, output])
    if output.exists():
        raise ValueError('Refusing to overwrite model diagnostics')
    manifest = json.loads((dataset/'manifest.json').read_text()); a = {}
    for name in ['returns','signal_index','security_key','eligible','dates']:
        spec = manifest['arrays'][name]; path = dataset/spec['path']
        if sha(path) != spec['sha256']:
            raise ValueError('Dataset changed since its freeze')
        a[name] = np.load(path, mmap_mode='r', allow_pickle=False)
    rows = []
    for receipt in sorted(results.glob('*/receipt.json')):
        proof = json.loads(receipt.read_text())
        if proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
            raise ValueError('Model/dataset mismatch')
        if max(proof['last_training_label_index'], proof['last_selection_label_index']) >= proof['first_prediction_signal_index']:
            raise ValueError('Chronological leakage in model receipt')
        for name in ['sample_ids.npy','scores.npy']:
            if sha(receipt.parent/name) != proof['artifacts'][name]:
                raise ValueError('Forecast hash mismatch')
        ids = np.load(receipt.parent/'sample_ids.npy'); scores = np.load(receipt.parent/'scores.npy')
        fold = proof['fold']; s = a['signal_index']; eligible = a['eligible']
        expected = np.flatnonzero((s >= fold['test_start']) & (s < fold['test_end']) & eligible)
        if not np.array_equal(ids, expected) or scores.shape != ids.shape:
            raise ValueError('Prediction membership changed after observing outcomes')
        daily = []
        for day in np.unique(s[ids]):
            selected = s[ids] == day; ix = ids[selected]
            metric = date_metrics(scores[selected], a['returns'][ix], a['security_key'][ix])
            daily.append(dict(date=str(a['dates'][day]), **metric))
        summary = {}
        for name in ['topk_raw_h5','observed_universe_raw_h5','topk_excess_h5','rank_ic']:
            values = [v[name] for v in daily if v[name] is not None]
            summary[name] = float(np.mean(values)) if values else None
        rows.append(dict(job=receipt.parent.name, objective=proof['config']['objective'],
            fold=fold['id'], seconds=proof['seconds'], best_epoch=proof['selection']['best_epoch'],
            calendar_sessions=fold['test_end']-fold['test_start'],
            missing_signal_sessions=[str(a['dates'][d]) for d in range(fold['test_start'],fold['test_end']) if d not in set(s[ids])],
            prediction_days=len(daily), scored_topk_days=sum(v['topk_raw_h5'] is not None for v in daily),
            summary=summary, daily=daily))
    result = dict(purpose='exploratory_model_diagnostic', contest_certified=False,
        threshold_evaluation_permitted=False, sharpe_computed=False,
        explanation='Means of overlapping five-session price-return labels, not realized daily NAV returns. Fees, sector/cap limits, actions and execution remain to be replayed.',
        dataset_manifest_sha256=sha(dataset/'manifest.json'), limitations=manifest['limitations'], jobs=rows)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps([dict(job=r['job'], **r['summary']) for r in rows]), flush=True)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--results', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); evaluate(args.dataset, args.results, args.output)

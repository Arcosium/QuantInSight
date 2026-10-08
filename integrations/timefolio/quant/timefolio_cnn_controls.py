"""Past-only ranking controls under the same CNN cohort and account frictions.

These controls diagnose whether an image model adds value. They are not CNN
models, and are not substitutes for the requested image-learning strategy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quant.timefolio_cnn_dataset import sha
from quant.timefolio_cnn_account import evaluate_daily, turnover_windows, write
from quant.timefolio_heatmap_locked_weighted_replay import replay


def control_scores(close, adv5, membership):
    close, adv5, membership = map(np.asarray, [close, adv5, membership])
    if close.ndim != 2 or adv5.shape != close.shape or membership.shape != close.shape:
        raise ValueError('Matching date-security arrays required')
    result = {name: np.full(close.shape, np.nan) for name in ['momentum20', 'reversal5', 'low_vol20', 'liquidity5']}
    for day in range(19, len(close)):
        window = close[day-19:day+1]
        valid = (np.isfinite(window).all(axis=0) & (window > 0).all(axis=0)
                 & np.isfinite(adv5[day]) & (adv5[day] > 0) & membership[day])
        prices = window[:, valid]
        result['momentum20'][day, valid] = np.log(prices[-1]/prices[0])
        result['reversal5'][day, valid] = -np.log(prices[-1]/prices[-6])
        result['low_vol20'][day, valid] = -np.diff(np.log(prices), axis=0).std(axis=0, ddof=1)
        result['liquidity5'][day, valid] = np.log(adv5[day, valid])
    return result


def run(dataset, panel_root, output):
    dataset, panel_root, output = map(Path, [dataset, panel_root, output])
    if output.exists():
        raise ValueError('Use a fresh control-results directory')
    manifest = json.loads((dataset/'manifest.json').read_text())
    panel_proof = json.loads((panel_root/'receipt.json').read_text())
    if (dataset/'RETIRED.json').exists() or panel_proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
        raise ValueError('Consistent matching dataset required')
    arrays = {}
    for name in ['daily_ohlcv', 'adv5_proxy', 'signal_index', 'security_key', 'eligible']:
        spec = manifest['arrays'][name]; path = (dataset/spec['path']).resolve()
        if not path.is_relative_to(dataset.resolve()) or sha(path) != spec['sha256']:
            raise ValueError('Control source data changed')
        arrays[name] = np.load(path, allow_pickle=False, mmap_mode='r')
    for name, digest in panel_proof['artifacts'].items():
        path = (panel_root/name).resolve()
        if not path.is_relative_to(panel_root.resolve()) or sha(path) != digest:
            raise ValueError('Control account panel changed')
    with np.load(panel_root/'panel.npz', allow_pickle=False) as z:
        panel = {name: z[name] for name in z.files}
    index = json.loads((panel_root/'index.json').read_text())
    close = arrays['daily_ohlcv'][:, :, 3]
    membership = np.zeros(close.shape, bool)
    membership[arrays['signal_index'], arrays['security_key']] = arrays['eligible']
    scores = control_scores(close, arrays['adv5_proxy'], membership)
    plan = dict(controls=list(scores), buffers=[0, 5], start='20240101', end=index['dates'][-1],
        matched_CNN_membership=True, learned_model=False, gross=.8, rebalance_sessions=5,
        top_n=12, target_weight=.08, daily_order_budget=10, rebalance_band=.005,
        source_sha256=sha(__file__), dataset_manifest_sha256=sha(dataset/'manifest.json'),
        panel_receipt_sha256=sha(panel_root/'receipt.json'), contest_certified=False)
    output.mkdir(parents=True); write(output/'plan.json', plan); summary = []
    for name, score in scores.items():
        for buffer in plan['buffers']:
            result = replay(panel, index, score.T, plan['start'], plan['end'],
                gross_schedule=np.full(len(index['dates']), .8), rebalance=5, top_n=12,
                weight=.08, gross=.8, max_orders=10, rank_buffer=buffer, rebalance_band=.005)
            evaluation = evaluate_daily(result['daily']); windows = turnover_windows(result['daily'])
            result['metrics']['whole_period_four_week_turnover_stop'] = result['metrics'].pop('four_week_turnover_stop')
            result['metrics']['sharpe'] = evaluation['pooled']['sharpe']
            case = dict(control=name, rank_buffer=buffer, **evaluation,
                metrics=result['metrics'], turnover_windows=windows, contest_certified=False,
                qualified_for_record=False, qualified_for_stop=False)
            result['evaluation'] = case; filename = f'{name}_buffer{buffer}.json'
            write(output/filename, result); summary.append(dict(file=filename, sha256=sha(output/filename), **case))
            write(output/'progress.json', dict(completed=len(summary), planned=8))
            print(json.dumps(dict(control=name, buffer=buffer, **evaluation['pooled'])), flush=True)
    write(output/'summary.json', dict(plan=plan, cases=summary, limitations=panel_proof['limitations']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--panel', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args(); run(args.dataset, args.panel, args.output)

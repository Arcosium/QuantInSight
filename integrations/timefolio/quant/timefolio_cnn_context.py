"""Causal cross-sectional and sector context rows for chart CNN experiments.

Every context cell uses observations through that date only. The cohort and
current sector snapshot remain explicitly exploratory historical proxies.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from quant.timefolio_cnn_dataset import sha

FEATURES = ['return5_rank', 'return20_rank', 'negative_volatility20_rank',
            'volume5_over20_rank', 'adv5_rank', 'sector_relative_return20',
            'cohort_median_return20', 'cohort_positive_return20_breadth']


def ranks(values):
    """Average ties, with finite singleton/constant cross sections at zero."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Finite cross-sectional ranks required')
    if len(values) < 2:
        return np.zeros(len(values))
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    begins = np.r_[0, counts.cumsum()[:-1]]
    return 2 * (begins[inverse] + (counts[inverse]-1)/2) / (len(values)-1) - 1


def context_rows(close, volume, adv5, membership, sectors):
    close, volume, adv5, membership, sectors = map(np.asarray, [close, volume, adv5, membership, sectors])
    if (close.ndim != 2 or any(v.shape != close.shape for v in [volume, adv5, membership])
            or sectors.shape != (close.shape[1],) or membership.dtype != bool):
        raise ValueError('Matching observed date-security arrays required')
    d, n = close.shape
    out = np.zeros((d, n, len(FEATURES)), np.float32)
    available = np.zeros((d, n), bool)
    for day in range(20, d):
        prices = close[day-20:day+1]
        volumes = volume[day-19:day+1]
        valid = (membership[day] & np.isfinite(prices).all(axis=0) & (prices > 0).all(axis=0)
                 & np.isfinite(volumes).all(axis=0) & (volumes >= 0).all(axis=0)
                 & (volumes.sum(axis=0) > 0) & np.isfinite(adv5[day]) & (adv5[day] > 0))
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        p, v = prices[:, ids], volumes[:, ids]
        r5, r20 = p[-1]/p[-6]-1, p[-1]/p[0]-1
        metrics = [r5, r20, -np.diff(np.log(p), axis=0).std(axis=0, ddof=1),
                   v[-5:].mean(axis=0)/v.mean(axis=0), adv5[day, ids]]
        for column, values in enumerate(metrics):
            out[day, ids, column] = ranks(values)
        for sector in np.unique(sectors[ids]):
            members = sectors[ids] == sector
            if sector < 0 or members.sum() < 2:
                continue
            out[day, ids[members], 5] = np.clip((r20[members]-np.median(r20[members]))/.25, -1, 1)
        out[day, ids, 6] = np.clip(np.median(r20)/.25, -1, 1)
        out[day, ids, 7] = 2*float(np.mean(r20 > 0))-1
        available[day, ids] = True
    return out, available


def build(dataset, account_panel, output):
    dataset, account_panel, output = map(lambda p: Path(p).resolve(), [dataset, account_panel, output])
    if output.exists():
        raise ValueError('Use a fresh context experiment directory')
    if (dataset/'RETIRED.json').exists():
        raise ValueError('Cannot derive images from a retired dataset')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('feature_rows', 8) != 8 or manifest.get('exploratory_ready') is not True:
        raise ValueError('An eight-row exploratory chart dataset is required')
    proof = json.loads((account_panel/'receipt.json').read_text())
    if proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
        raise ValueError('Context sectors must match the account dataset')
    for filename, digest in proof['artifacts'].items():
        p = (account_panel/filename).resolve()
        if not p.is_relative_to(account_panel) or sha(p) != digest:
            raise ValueError('Account input changed')
    arrays = {}
    for name, spec in manifest['arrays'].items():
        p = (dataset/spec['path']).resolve()
        if not p.is_relative_to(dataset) or sha(p) != spec['sha256']:
            raise ValueError('Parent array changed')
        arrays[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    index = json.loads((account_panel/'index.json').read_text())
    if index['codes'] != arrays['codes'].tolist() or index['dates'] != arrays['dates'].tolist():
        raise ValueError('Context identity axes differ')
    with np.load(account_panel/'panel.npz', allow_pickle=False) as z:
        sectors = z['sector']
    membership = np.zeros(arrays['daily_ohlcv'].shape[:2], bool)
    membership[arrays['signal_index'], arrays['security_key']] = arrays['eligible']
    context, available = context_rows(arrays['daily_ohlcv'][:, :, 3], arrays['daily_ohlcv'][:, :, 4],
                                       arrays['adv5_proxy'], membership, sectors)
    child = output/'dataset'; child.mkdir(parents=True)
    started = time.monotonic(); n, _, _, window = arrays['images'].shape
    images = np.lib.format.open_memmap(child/'images.npy', mode='w+', dtype=np.float32,
                                      shape=(n, 1, 16, window))
    observed, total = 0, 0
    for lo in range(0, n, 512):
        hi = min(lo+512, n)
        days = arrays['signal_index'][lo:hi, None] - np.arange(window-1, -1, -1)[None]
        keys = arrays['security_key'][lo:hi, None]
        if (days < 0).any():
            raise ValueError('Image window precedes available history')
        images[lo:hi, :, :8] = arrays['images'][lo:hi]
        images[lo:hi, 0, 8:] = context[days, keys].transpose(0, 2, 1)
        observed += int(available[days, keys].sum()); total += days.size
    images.flush(); del images
    for name, spec in manifest['arrays'].items():
        if name != 'images':
            os.link(dataset/spec['path'], child/spec['path'])
    for filename in ['cohort.json', 'price_reference_hashes.json']:
        if (dataset/filename).exists(): shutil.copy2(dataset/filename, child/filename)
    manifest.update(parent_manifest_sha256=sha(dataset/'manifest.json'), feature_rows=16,
        context_features=FEATURES, context_source_sha256=sha(__file__),
        context_sector_panel_receipt_sha256=sha(account_panel/'receipt.json'),
        context_available_window_fraction=observed/total,
        context_missing_policy='Neutral zero before20past sessions, outside that date membership, or incomplete history. Unknown/singleton sector residual is zero.',
        context_membership='Each context date uses that date eligible liquidity membership, not the later image-end membership.',
        limitations=manifest['limitations']+[
            'Current sector snapshot is used as exploratory historical image context; historical sector identities are not certified.',
            'Cohort return and breadth describe the frozen research cohort, not an official KRX market index.',
            'Foreign/institutional flow is not present in these context rows.'],
        derived_at=time.time(), context_build_seconds=time.monotonic()-started)
    manifest['arrays']['images'] = dict(path='images.npy', sha256=sha(child/'images.npy'))
    (child/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    panel_copy = output/'account_panel'; panel_copy.mkdir()
    for filename in proof['artifacts']:
        os.link(account_panel/filename, panel_copy/filename)
    proof.update(dataset_manifest_sha256=sha(child/'manifest.json'),
                 parent_panel_receipt_sha256=sha(account_panel/'receipt.json'),
                 reused_account_arrays_without_modification=True)
    (panel_copy/'receipt.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(rows=n, feature_rows=16, window=window,
                          context_available_window_fraction=observed/total)), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ['dataset','account-panel','output']:
        parser.add_argument('--'+key, required=True)
    args=parser.parse_args();build(args.dataset,args.account_panel,args.output)

"""Separate economic returns from cost/downside-aware CNN training utilities.

Future holding lows are labels, never input features. Their availability is
bounded by the existing purged exit index. Prediction membership is unchanged.
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


def utility_targets(daily, signal, keys, end, economic_returns, *, horizon, penalty):
    daily, signal, keys, end, returns = map(np.asarray, [daily, signal, keys, end, economic_returns])
    if (daily.ndim != 3 or daily.shape[2] != 5 or signal.ndim != 1
            or any(a.shape != signal.shape for a in [keys, end, returns])
            or not all(np.issubdtype(a.dtype, np.integer) for a in [signal, keys, end])
            or horizon < 1 or penalty not in (0., .5, 1.)
            or np.any(signal < 0) or np.any(signal >= len(daily))
            or np.any(keys < 0) or np.any(keys >= daily.shape[1])
            or np.any(end != signal+horizon+1)):
        raise ValueError('Registered holding-period label axes required')
    net, adverse = [np.full(len(signal), np.nan) for _ in range(2)]
    ids = np.flatnonzero(np.isfinite(returns) & (end < len(daily)))
    for lo in range(0, len(ids), 512):
        ix = ids[lo:lo+512]; entry = daily[signal[ix]+1, keys[ix], 0]
        exit_price = daily[end[ix], keys[ix], 0]
        if (not np.isfinite(entry).all() or not np.isfinite(exit_price).all()
                or (entry <= 0).any() or (exit_price <= 0).any()):
            raise ValueError('Economic label has invalid entry or exit prices')
        gross = exit_price/entry-1
        if not np.allclose(gross, returns[ix], rtol=2e-5, atol=2e-6):
            raise ValueError('Economic labels disagree with registered open-to-open prices')
        # Same multiplicative full-roundtrip convention as the cash-gate labels.
        net[ix] = (1+gross)*(1-.0005)*(1-.003)/((1+.0005)*(1+.001))-1
        holding_days = signal[ix, None]+np.arange(1, horizon+1)[None]
        lows = daily[holding_days, keys[ix, None], 2]
        valid = np.isfinite(lows).all(axis=1) & (lows > 0).all(axis=1)
        adverse[ix[valid]] = np.maximum(0., 1-lows[valid].min(axis=1)/entry[valid])
    utility = net.copy() if penalty == 0 else net-penalty*adverse
    return dict(net_holding_returns=net, maximum_adverse_excursion=adverse,
                training_utility=utility)


def build(dataset, account_panel, output, *, penalty):
    dataset, account_panel, output = map(lambda p: Path(p).resolve(), [dataset, account_panel, output])
    if output.exists() or (dataset/'RETIRED.json').exists():
        raise ValueError('A fresh output and a nonretired parent are required')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('training_target', 'returns') != 'returns' or manifest.get('exploratory_ready') is not True:
        raise ValueError('Derive utilities once from a registered economic-label dataset')
    arrays = {}
    for name, spec in manifest['arrays'].items():
        p = (dataset/spec['path']).resolve()
        if not p.is_relative_to(dataset) or sha(p) != spec['sha256']:
            raise ValueError('Parent dataset changed')
        arrays[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    proof = json.loads((account_panel/'receipt.json').read_text())
    if proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json'):
        raise ValueError('Account panel identity mismatch')
    for name, digest in proof['artifacts'].items():
        if Path(name).name != name or sha(account_panel/name) != digest:
            raise ValueError('Account input changed')
    labels = utility_targets(arrays['daily_ohlcv'], arrays['signal_index'], arrays['security_key'],
                             arrays['label_end_index'], arrays['returns'],
                             horizon=manifest['horizon'], penalty=penalty)
    child = output/'dataset'; child.mkdir(parents=True)
    for spec in manifest['arrays'].values():
        os.link(dataset/spec['path'], child/spec['path'])
    for name, array in labels.items():
        if name in manifest['arrays']:
            raise ValueError('Training label name already exists')
        np.save(child/(name+'.npy'), array, allow_pickle=False)
        manifest['arrays'][name] = dict(path=name+'.npy', sha256=sha(child/(name+'.npy')))
    for filename in ['cohort.json', 'price_reference_hashes.json']:
        if (dataset/filename).exists(): shutil.copy2(dataset/filename, child/filename)
    manifest.update(utility_parent_manifest_sha256=sha(dataset/'manifest.json'),
        training_target='training_utility', training_target_description='Net open-to-open return minus lambda times maximum adverse excursion before exit open',
        downside_penalty=penalty, finite_training_targets=int(np.isfinite(labels['training_utility']).sum()),
        utility_source_sha256=sha(__file__), utility_created_at=time.time(),
        economic_returns_preserved=True, prediction_membership_preserved=True,
        target_costs=dict(buy_fee=.001, sell_fee=.003, slippage_each_way=.0005),
        target_low_interval='signal+1 through signal+H; exit-day low is excluded',
        limitations=manifest['limitations']+['Downside utility is a training surrogate, not an economic or cash-gate return.'])
    (child/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    panel_copy = output/'account_panel'; panel_copy.mkdir()
    for filename in proof['artifacts']:
        os.link(account_panel/filename, panel_copy/filename)
    proof.update(dataset_manifest_sha256=sha(child/'manifest.json'),
                 utility_parent_panel_receipt_sha256=sha(account_panel/'receipt.json'),
                 reused_account_arrays_without_modification=True)
    (panel_copy/'receipt.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(penalty=penalty, rows=manifest['rows'],
                          finite_targets=manifest['finite_training_targets'])), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for name in ['dataset','account-panel','output']:parser.add_argument('--'+name, required=True)
    parser.add_argument('--penalty',type=float,choices=[0.,.5,1.],required=True)
    args=parser.parse_args();build(args.dataset,args.account_panel,args.output,penalty=args.penalty)

"""Market-only cash forecasts learned before CNN OOF baskets are available.

The target is a preselected liquid market basket, not the actual CNN account.
CNN scores and execution rules remain unchanged; final selection needs account
replay. Existing chronological OOF gate outputs are preserved as comparators.
"""
from pathlib import Path
import json

import numpy as np

from quant.timefolio_cnn_dataset import sha

FEATURES = ['eligible_median_return5', 'eligible_median_return19',
            'eligible_breadth_above_ma20', 'eligible_median_log_vol19']


def market_records(arrays, tradable, *, horizon, top_k=50):
    from quant.timefolio_cnn_gate import net_roundtrip
    close = arrays['daily_ohlcv'][:, :, 3]
    signal, keys = arrays['signal_index'], arrays['security_key']
    if tradable.shape != close.shape or tradable.dtype != bool:
        raise ValueError('Signal-date tradability on matching date/security axes required')
    records = []
    valid = arrays['eligible'] & tradable[signal, keys]
    for day in np.unique(signal[valid]):
        if day < 19:
            continue
        ids = np.flatnonzero(valid & (signal == day)); stock = keys[ids]
        if len(stock) < top_k or len(np.unique(stock)) != len(stock):
            continue
        c = close[day-19:day+1, stock]
        adv = arrays['adv5_proxy'][day, stock]
        if not np.isfinite(c).all() or (c <= 0).any() or not np.isfinite(adv).all():
            raise ValueError('Known market features cannot contain missing price/liquidity inputs')
        selected = np.lexsort((stock, -adv))[:top_k]
        x = [np.median(c[-1]/c[-6]-1), np.median(c[-1]/c[0]-1),
             np.mean(c[-1] > c.mean(axis=0)), np.median(np.diff(np.log(c), axis=0).std(axis=0, ddof=1))]
        target = arrays['returns'][ids[selected]]
        # Fix the basket before consulting any future price or label mask.
        y = float(net_roundtrip(target.mean(), horizon=horizon)) if np.isfinite(target).all() else None
        records.append(dict(signal=int(day), label_end=int(day+horizon+1), features=list(map(float, x)),
            net_excess=y, selected_security_keys=stock[selected].tolist(), observed_members=len(stock)))
    return records


def walk_market_gate(records, folds, *, days, minimum_rows=60, alpha=10.):
    if not records or minimum_rows < 2 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Registered market observations and positive ridge penalty required')
    s = np.array([r['signal'] for r in records], dtype=int)
    end = np.array([r['label_end'] for r in records], dtype=int)
    x = np.array([r['features'] for r in records], dtype=float)
    y = np.array([np.nan if r['net_excess'] is None else r['net_excess'] for r in records])
    if (x.shape != (len(s), len(FEATURES)) or not np.isfinite(x).all()
            or len(np.unique(s)) != len(s) or (s < 0).any() or (s >= days).any() or (end <= s).any()):
        raise ValueError('Unique causal market observations required')
    expected = np.full(days, np.nan); fits = []
    for fold in folds:
        cutoff = fold['test_start']; forecast = (s >= cutoff) & (s < fold['test_end'])
        if not forecast.any():
            continue
        use = (s < cutoff) & (end < cutoff) & np.isfinite(y)
        proof = dict(fold=fold['id'], fit_cutoff=cutoff, matured_dates=int(use.sum()),
                     cold_start=int(use.sum()) < minimum_rows)
        if proof['cold_start']:
            proof['fallback'] = 'always invested until60matured market basket observations'
        else:
            mean, scale = x[use].mean(0), np.maximum(x[use].std(0), 1e-8)
            z = (x[use]-mean)/scale; intercept = float(y[use].mean())
            beta = np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]), z.T@(y[use]-intercept))
            expected[s[forecast]] = ((x[forecast]-mean)/scale)@beta+intercept
            proof.update(last_label_end=int(end[use].max()), mean=mean.tolist(), scale=scale.tolist(),
                         coefficient=beta.tolist(), intercept=intercept)
        fits.append(proof)
    exposures = np.where(np.isnan(expected) | (expected > 0), .8, 0.)
    return expected, exposures, np.maximum(exposures, .2), fits


def build_series(dataset, panel_root, output):
    dataset, panel_root, output = map(Path, [dataset, panel_root, output])
    if output.exists():
        raise ValueError('Use a fresh market-series directory')
    manifest = json.loads((dataset/'manifest.json').read_text())
    panel_proof = json.loads((panel_root/'receipt.json').read_text())
    if (manifest['cutoff'] != '20260923' or panel_proof['dataset_manifest_sha256'] != sha(dataset/'manifest.json')
            or manifest.get('exploratory_ready') is not True):
        raise ValueError('Frozen matching development inputs required')
    arrays = {}
    for name in ['daily_ohlcv','adv5_proxy','signal_index','security_key','eligible','returns','dates','codes']:
        spec = manifest['arrays'][name]; p = (dataset/spec['path']).resolve()
        if not p.is_relative_to(dataset.resolve()) or sha(p) != spec['sha256']:
            raise ValueError('Market series input changed')
        arrays[name] = np.load(p, mmap_mode='r', allow_pickle=False)
    for name, digest in panel_proof['artifacts'].items():
        p = (panel_root/name).resolve()
        if not p.is_relative_to(panel_root.resolve()) or sha(p) != digest:
            raise ValueError('Tradability input changed')
    index = json.loads((panel_root/'index.json').read_text())
    if index['dates'] != arrays['dates'].tolist() or index['codes'] != arrays['codes'].tolist():
        raise ValueError('Market and account axes differ')
    with np.load(panel_root/'panel.npz', allow_pickle=False) as z:
        tradable = z['eligible'].T
    records = market_records(arrays, tradable, horizon=manifest['horizon'])
    expected, cash, floor, fits = walk_market_gate(records, manifest['folds'], days=len(arrays['dates']))
    output.mkdir(parents=True)
    (output/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
    (output/'fits.json').write_text(json.dumps(fits, indent=2, allow_nan=False)+'\n')
    np.savez(output/'market_gate.npz', expected_net=expected, cold_start=np.isnan(expected),
             gross_ridge_cash=cash, gross_ridge_floor20=floor)
    np.savez(output/'market_records.npz', signal_index=np.array([r['signal'] for r in records]),
        label_end=np.array([r['label_end'] for r in records]),
        features=np.array([r['features'] for r in records]),
        net_excess=np.array([np.nan if r['net_excess'] is None else r['net_excess'] for r in records]))
    proof = dict(dataset_manifest_sha256=sha(dataset/'manifest.json'), features=FEATURES,
        panel_receipt_sha256=sha(panel_root/'receipt.json'), top_k=50, ridge_alpha=10., minimum_rows=60,
        horizon=manifest['horizon'], target='signal-date Top50liquidity equal-weight market basket net return',
        learned_cnn_used_for_market_target=False, selected_basket_missing_return_invalidates_whole_target=True,
        market_target_is_not_actual_CNN_account=True, cold_start_folds=[f['fold'] for f in fits if f['cold_start']],
        observations=len(records), finite_labels=sum(r['net_excess'] is not None for r in records),
        source_sha256=sha(__file__), artifacts={p.name:sha(p) for p in output.iterdir()}, contest_certified=False)
    (output/'receipt.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(dict(observations=proof['observations'], finite_labels=proof['finite_labels'],
                         cold_start_folds=proof['cold_start_folds'])), flush=True)
    return proof


def derive_gate(original, market_series, output):
    original, market_series, output = map(Path, [original, market_series, output])
    if output.exists():
        raise ValueError('Preserve the original OOF cash gate')
    proofs = []
    for root in [original, market_series]:
        proof = json.loads((root/'receipt.json').read_text())
        for name, digest in proof['artifacts'].items():
            p = (root/name).resolve()
            if not p.is_relative_to(root.resolve()) or sha(p) != digest:
                raise ValueError('Gate input fingerprint changed')
        proofs.append(proof)
    if proofs[0]['dataset_manifest_sha256'] != proofs[1]['dataset_manifest_sha256']:
        raise ValueError('The CNN and market gate belong to different datasets')
    with np.load(original/'gate.npz', allow_pickle=False) as z:
        arrays = {name:z[name] for name in z.files if name.startswith('gross_')}
    with np.load(market_series/'market_gate.npz', allow_pickle=False) as z:
        for name in z.files:
            if z[name].shape != arrays['gross_always'].shape:
                raise ValueError('Market exposure date axis differs')
            arrays[name] = z[name]
    with np.load(market_series/'market_records.npz', allow_pickle=False) as z:
        arrays.update({name:z[name] for name in z.files})
    # Original OOF features/targets remain in the untouched parent directory.
    # The derived feature arrays and receipt both describe the market model.
    output.mkdir(parents=True)
    import os
    os.link(original/'scores.npy', output/'scores.npy')
    np.savez(output/'gate.npz', **arrays)
    proof = dict(**{k:v for k,v in proofs[0].items() if k not in ['artifacts','source_sha256','features','basket_dates']},
        features=FEATURES, source_sha256=sha(__file__), parent_gate_receipt_sha256=sha(original/'receipt.json'),
        market_series_receipt_sha256=sha(market_series/'receipt.json'),
        exposure_model='independent market ridge, not selected-CNN-basket ridge',
        original_OOF_gate_preserved_at=str(original), basket_dates=proofs[1]['observations'],
        market_cold_start_folds=proofs[1]['cold_start_folds'], market_target_is_not_actual_CNN_account=True,
        unchanged_exposures=['always','trend_floor20','cash'],
        artifacts={p.name:sha(p) for p in output.iterdir()})
    (output/'receipt.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof

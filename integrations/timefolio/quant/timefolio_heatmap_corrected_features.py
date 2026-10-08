"""Rebuild action-sensitive image inputs from the original frozen primitives.

This is a data correction, not a performance-selected feature transformation.
Old float32 feature arrays and old images must reproduce before corrected images
are accepted. Execution-price/volume arrays and historical admission remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_features import infer_actions, lag, rolling, rank, make_image
from quant.timefolio_heatmap_action_amendment import amend_actions
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE, DEST as MODEL_ROOT
from quant.timefolio_heatmap_rank_seeds import DEST as SEED_ROOT

DEST = SOURCE.with_name('20260929_corrected_features_v1')
FIELDS = ['factor', 'r1', 'r5', 'r20', 'vol20', 'adv5', 'adv20', 'v20',
          'eligible', 'rank_r5', 'rank_vol', 'rank_adv', 'sector_r5', 'market_r5']


def derive_features(p, split, coverage):
    """Same causal feature equations as the frozen original panel constructor."""
    split = np.asarray(split, dtype=float); coverage = np.asarray(coverage)
    if split.shape != p['close'].shape or coverage.shape != split.shape:
        raise ValueError('Matching security/date arrays required')
    if not np.isfinite(split).all() or np.any(split <= 0):
        raise ValueError('Positive finite action ratios required')
    out = {'factor': np.cumprod(split, axis=1)}
    adjusted = p['close'] * out['factor']
    for h in [1, 5, 20]: out[f'r{h}'] = adjusted / lag(adjusted, h) - 1
    out['vol20'] = rolling(out['r1'], 20, 'std')
    out['adv5'] = rolling(p['traded_value'], 5); out['adv20'] = rolling(p['traded_value'], 20)
    out['v20'] = rolling(p['v'], 20)
    basis = p['price_basis']; good = (p['regular'] == 1) & (p['close'] > 0)
    volume_coverage = p['v'] / np.maximum(p['krx_volume'], 1)
    basis_valid = np.isfinite(basis) & (basis > 0) & (volume_coverage >= .85) & (volume_coverage <= 1.02)
    recent_good = rolling((good & basis_valid).astype(float), 20) >= .95
    no_jump = rolling((np.abs(out['r1']) > .35).astype(float), 20) == 0
    eligible = good & basis_valid & recent_good & no_jump & (p['market_cap'] >= 1e11) & (out['adv5'] > 3e9)
    eligible &= np.isfinite(out['vol20']) & np.isfinite(out['r20'])
    liquidity = pd.DataFrame(np.where(eligible, out['adv20'], np.nan)).rank(axis=0, ascending=False, method='first').to_numpy()
    out['eligible'] = eligible & (liquidity <= 200) & (rolling(coverage, 10) >= .85)
    out['rank_r5'] = rank(np.where(eligible, out['r5'], np.nan))
    out['rank_vol'] = rank(np.where(eligible, out['vol20'], np.nan))
    out['rank_adv'] = rank(np.where(eligible, out['adv20'], np.nan))
    out['sector_r5'] = np.full(split.shape, np.nan)
    for sec in np.unique(p['sector']):
        members = p['sector'] == sec
        values = pd.DataFrame(np.where(eligible[members], out['r5'][members], np.nan)).mean(axis=0).to_numpy()
        out['sector_r5'][members] = values
    market = pd.DataFrame(np.where(eligible, out['r5'], np.nan)).mean(axis=0).to_numpy()
    out['market_r5'] = np.broadcast_to(market, split.shape).copy()
    return out


def raw_panel(root, ix):
    """Restore double-precision primitives used before original float32 storage."""
    raw = pd.read_parquet(root / 'daily_minutes.parquet')
    meta = pd.read_parquet(root / 'historical_krx.parquet')
    snapshot = json.loads((root / 'timefolio_sectors.json').read_text())
    sectors = {s['code']: int(s['sector']) for s in snapshot['securities']}
    codes = sorted(set(raw.code) & set(sectors))
    dates = json.loads((root / 'calendar.json').read_text())
    if codes != ix['codes'] or dates != ix['dates']: raise AssertionError('Original axes changed')
    merged = meta.merge(raw, on=['code', 'date'], how='left', validate='one_to_one')
    p = {key: merged.pivot(index='code', columns='date', values=key).reindex(index=codes, columns=dates).to_numpy(dtype=float)
         for key in ['o', 'c', 'v', 'regular', 'exec_price', 'exec_volume', 'exec_high', 'exec_low', 'exec_count',
                     'market_cap', 'traded_value', 'krx_close', 'listed_shares', 'krx_volume']}
    p['sector'] = np.array([sectors[c] for c in codes], dtype=int)
    basis = np.divide(p['krx_close'], p['c'], out=np.full(p['c'].shape, np.nan), where=p['c'] > 0)
    p['price_basis'] = basis
    for key in ['o', 'c', 'exec_price', 'exec_high', 'exec_low']: p[key] *= basis
    for key in ['v', 'exec_volume']: p[key] /= basis
    p['close'] = np.where(p['krx_close'] > 0, p['krx_close'], p['c'])
    p['split'] = infer_actions(basis, p['regular'] == 1, p['krx_close'])
    return p


def freeze():
    root = Path(json.loads((SOURCE / 'rule_amendment.json').read_text())['source_run'])
    parent = json.loads((SEED_ROOT / 'protocol.json').read_text())
    for filename, value in parent['hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value:
            raise AssertionError('Parent input changed')
    files = [Path(__file__), SOURCE / 'rule_amendment.json', SEED_ROOT / 'protocol.json',
             SOURCE / 'panel.npz', SOURCE / 'panel_index.json', SOURCE / 'samples.npz', MODEL_ROOT / 'images_raw.npy']
    files += [root / name for name in ['calendar.json', 'daily_minutes.parquet', 'historical_krx.parquet', 'timefolio_sectors.json', 'bars15.npy']]
    files += [Path(__file__).with_name('timefolio_heatmap_' + n + '.py') for n in ['features', 'action_amendment', 'execution_amendment', 'study', 'walkforward', 'data']]
    spec = dict(source_root=str(root), parent=parent, fields=FIELDS,
                correction='Reconstruct original double-precision primitive inputs and derive original arrays, requiring exact float32 reproduction. Override only the five verified action ratios, recompute dependent features, ranks, sector/market context and universe; all original equations stay fixed.',
                precision='Feature arrays use original float32 storage. Execution primitives, sector caps and trade_allowed are copied unchanged from frozen panel. Corrected split ratios use exact existing amended-ledger precision.',
                images='Reproduce every existing raw32x65 image from frozen panel/bars before encoding corrected arrays. Keep original image construction and sample ordering. Save new samples if eligibility changes and disclose coverage changes.',
                labels='No new labels or returns evaluated in this data preparation; downstream labels must use corrected cumulative ledger factors and purged dates.',
                scope='Known events only; other inferred actions, missing dividends, current GICS/survivor universe and queue limitations remain. No reserved September28 outcomes.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Feature correction protocol changed')
    if not path.exists():
        atomic_json(path, spec); frozen = DEST / 'frozen_source'; frozen.mkdir(exist_ok=True)
        for p in files:
            if p.suffix == '.py': shutil.copy2(p, frozen / p.name)
    return spec


def prepare():
    spec = freeze(); root = Path(spec['source_root']); frozen, ix, ci, di = context(SOURCE)
    if ix['dates'][-1] != '20260923': raise AssertionError('Study cutoff changed')
    primitives = raw_panel(root, ix); bars = np.load(root / 'bars15.npy', mmap_mode='r')
    coverage = np.isfinite(bars[..., 3]).mean(axis=2)
    baseline = derive_features(primitives, primitives['split'], coverage); reproduction = {}
    for field in FIELDS:
        value = baseline[field].astype(bool if field == 'eligible' else np.float32)
        identical = np.array_equal(value, frozen[field], equal_nan=True)
        reproduction[field] = identical
        if not identical:
            atomic_json(DEST / 'failed_baseline_features.json', reproduction)
            raise AssertionError('Original feature does not reproduce: ' + field)
    actions = spec['parent']['training_parent']['actions']
    amended, _ = amend_actions(primitives, ix, actions)
    corrected = derive_features(primitives, amended['split'], coverage)
    panel = dict(frozen)
    for field in FIELDS: panel[field] = corrected[field].astype(bool if field == 'eligible' else np.float32)
    panel['split'] = amend_actions(frozen, ix, actions)[0]['split']
    cj, dj = np.where(panel['eligible']); order = np.lexsort((cj, dj)); cj, dj = cj[order], dj[order]
    old_images = np.load(MODEL_ROOT / 'images_raw.npy', mmap_mode='r')
    if (DEST / 'image_audit.json').exists(): raise RuntimeError('Completed feature preparation is immutable')
    for name in ['panel.npz', 'samples.npz', 'images_raw.npy']:
        if (DEST / name).exists(): raise RuntimeError('Partial feature output exists; inspect before resuming')
    # Existing-image reproduction is deliberately complete, not a selected sample.
    for j, (c, d) in enumerate(zip(ci, di)):
        if not np.array_equal(make_image(frozen, bars, c, d), old_images[j]):
            raise AssertionError(f'Original image does not reproduce at sample {j}')
        if j and j % 10000 == 0: print(json.dumps(dict(original_images_reproduced=j)), flush=True)
    atomic_json(DEST / 'baseline_reproduction.json', dict(features=reproduction, images=len(ci), exact=True))
    np.savez_compressed(DEST / 'panel.npz', **panel)
    np.savez_compressed(DEST / 'samples.npz', ci=cj, di=dj); atomic_json(DEST / 'panel_index.json', ix)
    output = np.lib.format.open_memmap(DEST / 'images_raw.npy', mode='w+', dtype='uint8', shape=(len(cj), 32, 65))
    old_lookup = {(int(c), int(d)): j for j, (c, d) in enumerate(zip(ci, di))}
    row_changes = np.zeros(32, np.int64); changes = []; common = 0; max_difference = 0
    for j, (c, d) in enumerate(zip(cj, dj)):
        output[j] = make_image(panel, bars, c, d)
        old_index = old_lookup.get((int(c), int(d)))
        if old_index is not None:
            common += 1; before = old_images[old_index]; mask = output[j] != before
            if mask.any():
                difference = int(np.abs(output[j].astype(int) - before.astype(int)).max())
                row_changes += mask.sum(axis=1); max_difference = max(max_difference, difference)
                changes.append(dict(code=ix['codes'][int(c)], date=ix['dates'][int(d)], pixels=int(mask.sum()), maximum_difference=difference))
        if j and j % 10000 == 0: print(json.dumps(dict(corrected_images_encoded=j)), flush=True)
    output.flush()
    feature_changes = {}
    for field in FIELDS:
        old, new = np.asarray(frozen[field]), np.asarray(panel[field])
        equal = (old == new) | (np.isnan(old) & np.isnan(new))
        feature_changes[field] = int((~equal).sum())
    audit = dict(status='data correction only; no performance claim', original_samples=len(ci), corrected_samples=len(cj),
                 unchanged_sample_axes=bool(np.array_equal(ci, cj) and np.array_equal(di, dj)),
                 added_eligible_security_days=int((panel['eligible'] & ~frozen['eligible']).sum()),
                 removed_eligible_security_days=int((~panel['eligible'] & frozen['eligible']).sum()),
                 common_samples=common, changed_common_images=len(changes), changed_pixels=int(row_changes.sum()),
                 changed_pixels_by_row=row_changes.tolist(), maximum_pixel_difference=max_difference,
                 changed_feature_security_days=feature_changes, execution_and_admission_primitives_unchanged=True)
    for key in frozen:
        if key not in FIELDS + ['split'] and not np.array_equal(frozen[key], panel[key], equal_nan=True):
            raise AssertionError('Execution primitive changed: ' + key)
    atomic_json(DEST / 'changed_images.json', changes); atomic_json(DEST / 'image_audit.json', audit)
    paths = [DEST / n for n in ['panel.npz', 'panel_index.json', 'samples.npz', 'images_raw.npy']]
    atomic_json(DEST / 'data_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(audit), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'prepare']); a = ap.parse_args()
    if a.action == 'freeze': freeze()
    else: prepare()

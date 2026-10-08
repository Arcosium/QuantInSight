"""Same-date, leave-one-image-out peer context beside corrected stock images."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from quant.timefolio_heatmap_corrected_features import DEST as STOCK_ROOT
from quant.timefolio_heatmap_data import atomic_json
from quant.timefolio_heatmap_study import context
from quant.timefolio_heatmap_walkforward import SOURCE

DEST = SOURCE.with_name('20260929_peer_features_v1')
MASKED_ROWS = np.r_[0:14, 28:30]


def excluding_self_mean(block):
    """Exact integer sums make the rounded peer pixels order independent."""
    present = np.ones(block.shape, dtype=bool)
    present[:, MASKED_ROWS] = (block[:, 14] > 128)[:, None, :]
    values = block.astype(np.int64) * present
    numerator = values.sum(axis=0)[None] - values
    denominator = present.sum(axis=0)[None] - present.astype(np.int64)
    out = np.full(block.shape, 128., dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return np.rint(out).astype(np.uint8)


def peer_images(raw, ci, di, sectors):
    """Use >=3 other sector images, otherwise all other same-date images.

    This excludes the target image from the aggregation, not all indirect effects
    of its stock on existing cross-sectional rank features. No labels are read.
    """
    raw, ci, di, sectors = map(np.asarray, (raw, ci, di, sectors))
    n = len(raw)
    if (raw.ndim != 3 or raw.shape[1] != 32 or raw.dtype != np.uint8 or raw.shape[2] == 0
            or ci.shape != (n,) or di.shape != (n,) or sectors.ndim != 1
            or not np.issubdtype(ci.dtype, np.integer) or not np.issubdtype(di.dtype, np.integer)
            or not np.issubdtype(sectors.dtype, np.integer)):
        raise ValueError('uint8 N x32xT images and matching integer sample/sector vectors required')
    if (np.any(ci < 0) or np.any(ci >= len(sectors)) or np.any(di < 0)
            or len(np.unique(np.column_stack([ci, di]), axis=0)) != n):
        raise ValueError('Unique, nonnegative, valid stock/date samples required')
    result = np.empty_like(raw); counts = np.zeros(n, np.int32); sector_used = np.zeros(n, bool)
    for day in np.unique(di):
        ids = np.flatnonzero(di == day)
        result[ids] = excluding_self_mean(raw[ids]); counts[ids] = len(ids) - 1
        for sector in np.unique(sectors[ci[ids]]):
            local = ids[sectors[ci[ids]] == sector]
            if len(local) >= 4:
                result[local] = excluding_self_mean(raw[local])
                counts[local] = len(local) - 1; sector_used[local] = True
    return result, counts, sector_used


def freeze():
    manifests = [json.loads((STOCK_ROOT / 'protocol.json').read_text())['hashes'],
                 json.loads((STOCK_ROOT / 'data_hashes.json').read_text())]
    for manifest in manifests:
        for filename, value in manifest.items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != value:
                raise AssertionError('Corrected stock input changed')
    files = [Path(__file__), STOCK_ROOT / 'protocol.json', STOCK_ROOT / 'data_hashes.json',
             STOCK_ROOT / 'panel.npz', STOCK_ROOT / 'panel_index.json', STOCK_ROOT / 'samples.npz',
             STOCK_ROOT / 'images_raw.npy', Path(__file__).with_name('timefolio_heatmap_study.py'),
             Path(__file__).with_name('timefolio_heatmap_data.py')]
    spec = dict(source=str(STOCK_ROOT), signal_period=['20250908', '20260923'],
                context='Same-signal-date stock images only. Mean of at least3 other same-sector samples; otherwise other market samples that date. Target image subtracted. No forward labels, outcome weights or pooled future dates.',
                pixels='All32 rows averaged in original uint8 space, exact int64 sums, round-to-even. Rows0:14 and28:30 exclude slots where peer coverage row14<=128. No observations yield neutral128. Other rows, including coverage, averaged directly.',
                control='Original corrected stock channel remains unchanged; blank control suppresses the context channel to exactly0 after normalization inside the network.',
                limitations='Mean encoded features is not an investible sector price series. Existing cross-sectional ranks/context within source images may indirectly include the target stock. Static GICS and survivor universe remain.',
                hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    DEST.mkdir(exist_ok=True); path = DEST / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != spec: raise RuntimeError('Peer feature protocol changed')
    if not path.exists():
        atomic_json(path, spec); folder = DEST / 'frozen_source'; folder.mkdir(exist_ok=True)
        for i, p in enumerate(files):
            if p.suffix == '.py': shutil.copy2(p, folder / (f'{i:02d}_' + p.name))
    return spec


def prepare():
    freeze(); p, ix, ci, di = context(STOCK_ROOT)
    if ix['dates'][-1] != '20260923': raise AssertionError('Unexpected date coverage')
    raw = np.load(STOCK_ROOT / 'images_raw.npy', mmap_mode='r')
    output = DEST / 'images_peer.npy'
    if output.exists(): raise RuntimeError('Do not overwrite prepared peer features')
    images, counts, used = peer_images(raw, ci, di, p['sector'])
    np.save(output, images); np.savez_compressed(DEST / 'peer_membership.npz', counts=counts, sector_used=used)
    summary = dict(samples=len(ci), shape=list(images.shape), sector_context_samples=int(used.sum()),
                   market_fallback_samples=int((~used).sum()), minimum_peers=int(counts.min()),
                   median_peers=float(np.median(counts)), maximum_peers=int(counts.max()),
                   raw_channel_unchanged=True, selected_without_labels=True)
    atomic_json(DEST / 'input_summary.json', summary)
    files = [output, DEST / 'peer_membership.npz']
    atomic_json(DEST / 'data_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['freeze', 'prepare']); args = ap.parse_args()
    if args.action == 'freeze': freeze()
    else: prepare()

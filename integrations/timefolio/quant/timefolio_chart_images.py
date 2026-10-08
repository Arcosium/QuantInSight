"""Causal OHLCV rasterization using frozen data, without chart labels or outcomes."""
import argparse
import fcntl
import json
from pathlib import Path
import numpy as np
from quant.timefolio_chart_lab import VIEWS, SHAPE
from quant.timefolio_heatmap_gpu_worker import digest, write


def history(daily, intraday, split, security, day, view):
    if view not in VIEWS: raise ValueError('Unknown registered chart')
    length = 5 if view == 'intraday5' else 60 if view in ['candle60', 'multi20_60'] else 20
    first = max(0, day - length + 1)
    # Factors are constructed only through the signal close. No later action
    # can change this image. All OHLC prices use the signal-day share basis.
    ratios = np.asarray(split[security, first:day + 1], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError('Invalid historical share factor')
    factors = np.cumprod(ratios); factors /= factors[-1]
    source = intraday[security, first:day + 1] if view == 'intraday5' else daily[security, first:day + 1]
    out = np.full((length, 13, 5) if view == 'intraday5' else (length, 5), np.nan)
    values = np.array(source, dtype=float, copy=True)
    scale = factors[:, None] if view == 'intraday5' else factors
    values[..., :4] *= scale[..., None]; values[..., 4] /= scale
    out[-len(values):] = values
    return out.reshape(-1, 5)


def line(image, x0, y0, x1, y1, value=0):
    n = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xx = np.rint(np.linspace(x0, x1, n)).astype(int)
    yy = np.rint(np.linspace(y0, y1, n)).astype(int)
    image[yy, xx] = value


def raster(bars, style='candle', height=64, width=192):
    """White background, dark price marks, grey volume; missing bars stay blank."""
    a = np.asarray(bars, dtype=float)
    if a.ndim != 2 or a.shape[1] != 5: raise ValueError('OHLCV required')
    image = np.full((height, width), 255, np.uint8)
    valid = np.isfinite(a).all(axis=1) & (a[:, :4] > 0).all(axis=1) & (a[:, 4] >= 0)
    valid &= (a[:, 1] >= a[:, [0, 2, 3]].max(axis=1)) & (a[:, 2] <= a[:, [0, 1, 3]].min(axis=1))
    if not valid.any(): return image
    # The price-only line uses close bounds, so hidden high/low values are not
    # encoded accidentally through its vertical normalization.
    values = a[valid, 3] if style == 'line' else a[valid, :4]
    low, high = float(values.min()), float(values.max())
    span = max(high - low, max(abs(high), 1.) * 1e-6)
    top, bottom = 2, height * 3 // 4 - 3
    x = np.rint(np.linspace(3, width - 4, len(a))).astype(int)
    def y(price): return int(np.clip(round(bottom - (price - low) / span * (bottom - top)), top, bottom))
    volume_bottom = height - 3; volume_top = height * 3 // 4 + 1
    max_volume = max(float(a[valid, 4].max()), 1.)
    for i in np.flatnonzero(valid):
        xx = x[i]; o, hi, lo, c, vol = a[i]
        if style == 'line':
            image[y(c), xx] = 0
            if i > 0 and valid[i - 1]: line(image, x[i - 1], y(a[i - 1, 3]), xx, y(c))
        else:
            image[y(hi):y(lo) + 1, xx] = 0
            if style == 'ohlc':
                image[y(o), xx - 1:xx + 1] = 0; image[y(c), xx:xx + 2] = 0
            elif style == 'candle':
                ya, yb = sorted((y(o), y(c)))
                image[ya:yb + 1, xx - 1:xx + 2] = 0
                if c > o and yb - ya > 1: image[ya + 1:yb, xx] = 255
            else: raise ValueError('Unknown chart style')
        if vol > 0:
            h = max(1, round(vol / max_volume * (volume_bottom - volume_top + 1)))
            image[volume_bottom - h + 1:volume_bottom + 1, xx - 1:xx + 2] = 96
    return image


def render(daily, intraday, split, security, day, view):
    bars = history(daily, intraday, split, security, day, view)
    if view == 'multi20_60':
        return np.concatenate([raster(bars[-20:], height=32), raster(bars, height=32)])
    style = 'ohlc' if view == 'ohlc20' else 'line' if view == 'line20' else 'candle'
    return raster(bars, style=style)


def prepare(root, view):
    root = Path(root); path = root / f'images_{view}.npy'; receipt = root / f'images_{view}.json'
    if view not in VIEWS: raise ValueError('Unregistered image view')
    with (root / f'images_{view}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = ['chart_daily.npy', 'chart_intraday.npy', 'chart_split.npy', 'samples.npz']
        hashes = {p: digest(root / p) for p in inputs}
        if receipt.exists():
            old = json.loads(receipt.read_text())
            assert old['inputs'] == hashes and digest(path) == old['sha256']
            return path
        if path.exists(): raise RuntimeError('Incomplete image cache requires review')
        daily, intra, split = [np.load(root / p, mmap_mode='r') for p in inputs[:3]]
        samples = np.load(root / 'samples.npz'); ci, di = samples['ci'], samples['di']
        out = np.lib.format.open_memmap(path, mode='w+', dtype='uint8', shape=(len(ci), *SHAPE))
        for row, (security, day) in enumerate(zip(ci, di)):
            out[row] = render(daily, intra, split, int(security), int(day), view)
        out.flush(); del out
        write(receipt, dict(view=view, shape=[len(ci), *SHAPE], inputs=hashes,
            sha256=digest(path), input_time='signal_close', future_inputs=False))
        return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--view', choices=VIEWS, required=True); args = p.parse_args()
    print(prepare(args.root, args.view), flush=True)

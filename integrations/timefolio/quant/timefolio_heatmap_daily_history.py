"""Put archived daily summaries on their own dates in a five-day heatmap.

All columns are an end-of-signal-day representation, not intraday predictions.
The original snapshot and every non-summary row are preserved as controls.
"""
from __future__ import annotations

import numpy as np


DAILY_ROWS = np.array([11, 12, 15, *range(16, 28)], dtype=int)
FIELDS = ('v', 'v20', 'adv5', 'rank_adv', 'r1', 'r5', 'r20', 'vol20',
          'rank_r5', 'rank_vol', 'sector_r5', 'market_r5', 'market_cap', 'sector_cap')
MODES = ('snapshot', 'history', 'history_reverse')


def daily_values(panel, codes, days):
    """Original daily equations, before uint8 quantization; days are N by5."""
    f = {key: np.asarray(panel[key])[codes[:, None], days] for key in FIELDS}
    values = np.empty((len(codes), len(DAILY_ROWS), days.shape[1]), np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        values[:, 0] = np.log((f['v'] + 1) / (f['v20'] + 1)) / 2
        values[:, 1] = np.log10(np.maximum(f['adv5'], 1) / 3e9) / 3
        values[:, 2] = f['rank_adv'] * 2 - 1
        for row, key, scale in [(3, 'r1', .1), (4, 'r5', .2), (5, 'r20', .4),
                                (6, 'vol20', .05), (9, 'sector_r5', .2), (10, 'market_r5', .2)]:
            values[:, row] = f[key] / scale
        values[:, 7] = f['rank_r5'] * 2 - 1
        values[:, 8] = f['rank_vol'] * 2 - 1
        values[:, 11] = np.log10(np.maximum(f['market_cap'], 1) / 1e11) / 4
        values[:, 12] = np.where(f['market_cap'] < 1e12, 1, -1)
        values[:, 13] = f['sector_cap']
        values[:, 14] = 8e7 / np.maximum(f['adv5'] * .05, 1)
    return values


def quantize(values):
    return np.rint((np.clip(np.nan_to_num(values), -1, 1) + 1) * 127.5).astype(np.uint8)


def encode_daily_history(raw, panel, codes, days, mode):
    """Preserve the latest13 columns; reverse only the four older day blocks."""
    codes, days = np.asarray(codes), np.asarray(days)
    if mode not in MODES:
        raise ValueError('Unknown daily-summary representation')
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[1:] != (32, 65):
        raise ValueError('Archived uint8 five-day32x65 images required')
    if (codes.ndim != 1 or days.shape != codes.shape or len(codes) != len(raw)
            or not np.issubdtype(codes.dtype, np.integer) or not np.issubdtype(days.dtype, np.integer)):
        raise ValueError('Matching integer security and signal-date indices required')
    shape = np.asarray(panel['market_cap']).shape
    if (len(shape) != 2 or any(np.asarray(panel[key]).shape != shape for key in FIELDS)
            or np.any(codes < 0) or np.any(codes >= shape[0])
            or np.any(days < 4) or np.any(days >= shape[1])):
        raise ValueError('Daily panel axes and complete five-day windows required')
    day_grid = days[:, None] + np.arange(-4, 1)
    history = quantize(daily_values(panel, codes, day_grid))
    snapshot = np.repeat(history[:, :, -1:], 65, axis=2)
    if not np.array_equal(raw[:, DAILY_ROWS], snapshot):
        raise AssertionError('Original daily snapshot does not reproduce exactly')
    out = raw.copy()
    if mode != 'snapshot':
        order = [3, 2, 1, 0, 4] if mode == 'history_reverse' else [0, 1, 2, 3, 4]
        out[:, DAILY_ROWS] = np.repeat(history[:, :, order], 13, axis=2)
    return out

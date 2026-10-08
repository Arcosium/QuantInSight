"""Extend daily context while preserving all existing intraday heatmap rows."""
import numpy as np
from quant.timefolio_heatmap_daily_history import DAILY_ROWS, FIELDS, daily_values, quantize

STRIDES = (1, 2, 4)
MASKS = ('constant', 'available')


def encode(history, panel, codes, days, stride, mask):
    codes, days = np.asarray(codes), np.asarray(days)
    if type(stride) is not int or stride not in STRIDES or mask not in MASKS:
        raise ValueError('Registered context stride and mask required')
    if history.dtype != np.uint8 or history.ndim != 3 or history.shape[1:] != (32, 65):
        raise ValueError('Archived uint8 Nx32x65 history images required')
    if codes.ndim != 1 or days.shape != codes.shape or len(codes) != len(history):
        raise ValueError('Matching sample axes required')
    if not np.issubdtype(codes.dtype, np.integer) or not np.issubdtype(days.dtype, np.integer):
        raise ValueError('Integer sample axes required')
    shape = np.asarray(panel['market_cap']).shape
    if len(shape) != 2 or any(np.asarray(panel[k]).shape != shape for k in FIELDS):
        raise ValueError('Matching daily panel axes required')
    if np.any(codes < 0) or np.any(codes >= shape[0]) or np.any(days < 4 * stride) or np.any(days >= shape[1]):
        raise ValueError('Context must remain inside the archived past')
    daily = daily_values(panel, codes, days[:, None] + np.arange(-4, 1) * stride)
    out = history.copy()
    out[:, DAILY_ROWS] = np.repeat(quantize(daily), 13, axis=2)
    extra = np.full_like(out, 255)
    if mask == 'available':
        extra[:, DAILY_ROWS] = np.repeat(np.isfinite(daily).astype(np.uint8) * 255, 13, axis=2)
    # Latest-day summaries and all17 intraday/time rows are exactly preserved.
    if not np.array_equal(out[:, DAILY_ROWS, -13:], history[:, DAILY_ROWS, -13:]):
        raise AssertionError('Latest daily context does not reproduce')
    return np.stack([out, extra], axis=1)
